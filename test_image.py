import cv2
from trt_detector import TRTDetector

model = TRTDetector(
    engine_path="model/best.engine",
    input_size=640,
    conf_thres=0.25,
    iou_thres=0.45
)

image = cv2.imread("test.jpg")

if image is None:
    raise RuntimeError("Cannot read test.jpg")

detections = model(image)

print("Detections:")
for det in detections:
    print(det)

result = model.draw(image, detections)
cv2.imwrite("result.jpg", result)

print("Saved result.jpg")
