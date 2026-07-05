import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTDetector:
    def __init__(
        self,
        engine_path,
        input_size=640,
        conf_thres=0.25,
        iou_thres=0.45
    ):
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.logger = trt.Logger(trt.Logger.WARNING)

        # =========================
        # 1. 載入 TensorRT engine
        # =========================
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine: {}".format(
                    engine_path
                )
            )

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError(
                "Failed to create TensorRT execution context"
            )

        # CUDA Stream
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []

        # =========================
        # 2. TensorRT 10 I/O API
        # =========================
        for i in range(self.engine.num_io_tensors):

            tensor_name = self.engine.get_tensor_name(i)

            tensor_shape = tuple(
                self.engine.get_tensor_shape(tensor_name)
            )

            tensor_dtype = trt.nptype(
                self.engine.get_tensor_dtype(tensor_name)
            )

            tensor_mode = self.engine.get_tensor_mode(
                tensor_name
            )

            # 目前此版本假設固定 shape engine
            if any(dim < 0 for dim in tensor_shape):
                raise RuntimeError(
                    "Dynamic shape detected: {} shape={}".format(
                        tensor_name,
                        tensor_shape
                    )
                )

            size = trt.volume(tensor_shape)

            host_mem = cuda.pagelocked_empty(
                size,
                tensor_dtype
            )

            device_mem = cuda.mem_alloc(
                host_mem.nbytes
            )

            tensor_info = {
                "name": tensor_name,
                "host": host_mem,
                "device": device_mem,
                "shape": tensor_shape,
                "dtype": tensor_dtype
            }

            if tensor_mode == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)

            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                self.outputs.append(tensor_info)

            # TensorRT 10:
            # 告訴 context 每個 tensor 對應的 GPU buffer address
            self.context.set_tensor_address(
                tensor_name,
                int(device_mem)
            )

        if len(self.inputs) == 0:
            raise RuntimeError(
                "No input tensor found in engine"
            )

        if len(self.outputs) == 0:
            raise RuntimeError(
                "No output tensor found in engine"
            )

        print("[TRTDetector] Engine loaded.")

        print("[TRTDetector] Inputs:")
        for tensor in self.inputs:
            print(
                "  name={}, shape={}, dtype={}".format(
                    tensor["name"],
                    tensor["shape"],
                    tensor["dtype"]
                )
            )

        print("[TRTDetector] Outputs:")
        for tensor in self.outputs:
            print(
                "  name={}, shape={}, dtype={}".format(
                    tensor["name"],
                    tensor["shape"],
                    tensor["dtype"]
                )
            )

    # ==================================================
    # Letterbox
    # ==================================================
    def letterbox(self, image):

        h, w = image.shape[:2]

        scale = min(
            self.input_size / h,
            self.input_size / w
        )

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (
                self.input_size,
                self.input_size,
                3
            ),
            114,
            dtype=np.uint8
        )

        pad_x = (
            self.input_size - new_w
        ) // 2

        pad_y = (
            self.input_size - new_h
        ) // 2

        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w
        ] = resized

        return canvas, scale, pad_x, pad_y

    # ==================================================
    # Preprocess
    # ==================================================
    def preprocess(self, frame):

        img, scale, pad_x, pad_y = self.letterbox(
            frame
        )

        # OpenCV BGR -> RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # uint8 -> float32, normalize
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(
            img,
            (2, 0, 1)
        )

        # CHW -> NCHW
        img = np.expand_dims(
            img,
            axis=0
        )

        img = np.ascontiguousarray(
            img
        )

        return img, scale, pad_x, pad_y

    # ==================================================
    # TensorRT inference
    # ==================================================
    def infer_raw(self, input_tensor):

        input_info = self.inputs[0]

        # numpy -> pinned host memory
        np.copyto(
            input_info["host"],
            input_tensor.ravel()
        )

        # CPU -> GPU
        cuda.memcpy_htod_async(
            input_info["device"],
            input_info["host"],
            self.stream
        )

        # TensorRT 10:
        # execute_async_v3
        success = self.context.execute_async_v3(
            stream_handle=self.stream.handle
        )

        if not success:
            raise RuntimeError(
                "TensorRT execute_async_v3 failed"
            )

        # GPU -> CPU
        for output in self.outputs:

            cuda.memcpy_dtoh_async(
                output["host"],
                output["device"],
                self.stream
            )

        self.stream.synchronize()

        output_arrays = []

        for output in self.outputs:

            output_array = np.array(
                output["host"]
            )

            output_array = output_array.reshape(
                output["shape"]
            )

            output_arrays.append(
                output_array
            )

        return output_arrays

    # ==================================================
    # YOLO11 single-class postprocess
    # ==================================================
    def postprocess(
        self,
        outputs,
        scale,
        pad_x,
        pad_y,
        original_shape
    ):

        output = outputs[0]

        # 例如：
        # (1, 5, 8400)
        if output.ndim == 3:
            pred = output[0]
        else:
            pred = output

        # (5, 8400)
        # ->
        # (8400, 5)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        # 確認 YOLO11 單類別輸出
        if pred.shape[1] != 5:
            raise RuntimeError(
                "Unexpected YOLO output shape: {}. "
                "Expected single-class output with 5 values "
                "[x, y, w, h, confidence].".format(
                    pred.shape
                )
            )

        boxes = []
        scores = []
        class_ids = []

        h0, w0 = original_shape[:2]

        for det in pred:

            x = float(det[0])
            y = float(det[1])
            w = float(det[2])
            h = float(det[3])

            conf = float(det[4])

            if conf < self.conf_thres:
                continue

            # xywh -> xyxy
            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0

            # 還原 letterbox
            x1 = (
                x1 - pad_x
            ) / scale

            y1 = (
                y1 - pad_y
            ) / scale

            x2 = (
                x2 - pad_x
            ) / scale

            y2 = (
                y2 - pad_y
            ) / scale

            # 限制在原始影像範圍
            x1 = max(
                0,
                min(w0 - 1, x1)
            )

            y1 = max(
                0,
                min(h0 - 1, y1)
            )

            x2 = max(
                0,
                min(w0 - 1, x2)
            )

            y2 = max(
                0,
                min(h0 - 1, y2)
            )

            boxes.append([
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ])

            scores.append(conf)

            # 單類別
            class_ids.append(0)

        keep = self.nms(
            boxes,
            scores
        )

        detections = []

        for i in keep:

            detections.append({
                "box": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i]
            })

        return detections

    # ==================================================
    # NMS
    # ==================================================
    def nms(self, boxes, scores):

        if len(boxes) == 0:
            return []

        boxes_xywh = []

        for x1, y1, x2, y2 in boxes:

            boxes_xywh.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            self.conf_thres,
            self.iou_thres
        )

        if len(indices) == 0:
            return []

        return np.array(
            indices
        ).reshape(-1).tolist()

    # ==================================================
    # Detector call
    # ==================================================
    def __call__(self, frame):

        (
            input_tensor,
            scale,
            pad_x,
            pad_y
        ) = self.preprocess(frame)

        outputs = self.infer_raw(
            input_tensor
        )

        detections = self.postprocess(
            outputs,
            scale,
            pad_x,
            pad_y,
            frame.shape
        )

        return detections

    # ==================================================
    # Draw
    # ==================================================
    def draw(
        self,
        frame,
        detections,
        class_names=None
    ):

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

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = "{} {:.2f}".format(
                label,
                score
            )

            cv2.putText(
                frame,
                text,
                (
                    x1,
                    max(y1 - 10, 0)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame
