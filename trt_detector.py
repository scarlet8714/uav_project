import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTDetector:
    def __init__(self, engine_path, input_size=640, conf_thres=0.25, iou_thres=0.45):
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            binding_shape = self.engine.get_binding_shape(binding)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(binding_shape)

            host_mem = cuda.pagelocked_empty(size, binding_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    "name": binding,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": binding_shape,
                    "dtype": binding_dtype
                })
            else:
                self.outputs.append({
                    "name": binding,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": binding_shape,
                    "dtype": binding_dtype
                })

        print("[TRTDetector] Engine loaded.")
        print("[TRTDetector] Inputs:", self.inputs)
        print("[TRTDetector] Outputs:", self.outputs)

    def letterbox(self, image):
        h, w = image.shape[:2]
        new_shape = (self.input_size, self.input_size)

        scale = min(new_shape[0] / h, new_shape[1] / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3),
                         114, dtype=np.uint8)

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return canvas, scale, pad_x, pad_y

    def preprocess(self, frame):
        img, scale, pad_x, pad_y = self.letterbox(frame)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img, scale, pad_x, pad_y

    def infer_raw(self, input_tensor):
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())

        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output["host"],
                output["device"],
                self.stream
            )

        self.stream.synchronize()

        output_arrays = []
        for output in self.outputs:
            output_array = np.array(output["host"])
            output_array = output_array.reshape(output["shape"])
            output_arrays.append(output_array)

        return output_arrays

    def postprocess(self, outputs, scale, pad_x, pad_y, original_shape):
        output = outputs[0]

        # 常見 YOLOv8 detect output: (1, 5, 8400), (1, 6, 8400), (1, 84, 8400)
        if len(output.shape) == 3:
            pred = output[0]
        else:
            pred = output

        # 如果是 (channels, candidates)，轉成 (candidates, channels)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes = []
        scores = []
        class_ids = []

        h0, w0 = original_shape[:2]

        for det in pred:
            if len(det) < 5:
                continue

            x, y, w, h = det[0], det[1], det[2], det[3]

            # 情況 1：單類別，格式可能是 x,y,w,h,conf
            if len(det) == 5:
                conf = det[4]
                class_id = 0

            # 情況 2：格式可能是 x,y,w,h,obj,class_score...
            elif len(det) == 6:
                # 對單類別模型常見
                conf = det[4] * det[5]
                class_id = 0

            # 情況 3：多類別 YOLOv8 常見 x,y,w,h,class_scores...
            else:
                class_scores = det[4:]
                class_id = int(np.argmax(class_scores))
                conf = float(class_scores[class_id])

            if conf < self.conf_thres:
                continue

            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0

            # 還原 letterbox
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(conf))
            class_ids.append(class_id)

        keep = self.nms(boxes, scores)

        detections = []
        for i in keep:
            detections.append({
                "box": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i]
            })

        return detections

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        boxes_xywh = []
        for x1, y1, x2, y2 in boxes:
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            self.conf_thres,
            self.iou_thres
        )

        if len(indices) == 0:
            return []

        return indices.flatten().tolist()

    def __call__(self, frame):
        input_tensor, scale, pad_x, pad_y = self.preprocess(frame)
        outputs = self.infer_raw(input_tensor)
        detections = self.postprocess(
            outputs, scale, pad_x, pad_y, frame.shape)
        return detections

    def draw(self, frame, detections, class_names=None):
        if class_names is None:
            class_names = ["marker"]

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            class_id = det["class_id"]

            if class_id < len(class_names):
                label = class_names[class_id]
            else:
                label = str(class_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = "{} {:.2f}".format(label, score)
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame
