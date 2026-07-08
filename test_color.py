import cv2
import numpy as np

IMAGE_PATH = "car400.png"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

# 放大方便觀察
scale = 6
image_big = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_NEAREST,
)

hsv = cv2.cvtColor(image_big, cv2.COLOR_BGR2HSV)


def nothing(x):
    pass


cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)

# 初始值先用我們目前討論的紫紅色範圍
cv2.createTrackbar("H Min", "Controls", 105, 179, nothing)
cv2.createTrackbar("H Max", "Controls", 155, 179, nothing)

cv2.createTrackbar("S Min", "Controls", 35, 255, nothing)
cv2.createTrackbar("S Max", "Controls", 255, 255, nothing)

cv2.createTrackbar("V Min", "Controls", 15, 255, nothing)
cv2.createTrackbar("V Max", "Controls", 255, 255, nothing)


while True:
    h_min = cv2.getTrackbarPos("H Min", "Controls")
    h_max = cv2.getTrackbarPos("H Max", "Controls")

    s_min = cv2.getTrackbarPos("S Min", "Controls")
    s_max = cv2.getTrackbarPos("S Max", "Controls")

    v_min = cv2.getTrackbarPos("V Min", "Controls")
    v_max = cv2.getTrackbarPos("V Max", "Controls")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(
        image_big,
        image_big,
        mask=mask,
    )

    cv2.imshow("Original", image_big)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        break

cv2.destroyAllWindows()
