import numpy as np
import cv2


img = cv2.imread("./temp/img.jpg")
# img = cv2.resize(img, (720, 600))
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lo = np.asarray([11, 19, 64])
hi = np.asarray([105, 66, 255])


def nop(x):
  pass


cv2.namedWindow("mask")
cv2.createTrackbar("H_lo", "mask", lo[0], 255, nop)
cv2.createTrackbar("S_lo", "mask", lo[1], 255, nop)
cv2.createTrackbar("V_lo", "mask", lo[2], 255, nop)
cv2.createTrackbar("H_hi", "mask", hi[0], 255, nop)
cv2.createTrackbar("S_hi", "mask", hi[1], 255, nop)
cv2.createTrackbar("V_hi", "mask", hi[2], 255, nop)


show_boxes = False

while 1:
  mask = cv2.bitwise_not(cv2.inRange(hsv, lo, hi))
  # blur =cv2.GaussianBlur(mask, (5, 5), 1)
  # canny = cv2.Canny(blur, 10, 50)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

  # get absolute difference between dilate and thresh
  diff = cv2.absdiff(dilate, mask)
  # edges = 255 - diff
  edges = diff

  lo[0] = cv2.getTrackbarPos("H_lo", "mask")
  lo[1] = cv2.getTrackbarPos("S_lo", "mask")
  lo[2] = cv2.getTrackbarPos("V_lo", "mask")
  hi[0] = cv2.getTrackbarPos("H_hi", "mask")
  hi[1] = cv2.getTrackbarPos("S_hi", "mask")
  hi[2] = cv2.getTrackbarPos("V_hi", "mask")
  cv2.imshow("mask",cv2.resize( mask, (720, 500)))
  # cv2.drawContours(roi, rect_cntr, -1, (0, 0, 255), 2)
  # cv2.imshow("hsv",cv2.resize( hsv, (720, 500)))
  # cv2.imshow("blur",cv2.resize(blur , (720, 500)))
  # cv2.imshow("edges",cv2.resize( edges, (720, 500)))

  if show_boxes:
    roi = cv2.copyTo(img, mask=img)
    for i in cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
      eps = 0.05 * cv2.arcLength(i, True)
      approx = cv2.approxPolyDP(i, eps, True)
      if len(approx) == 4:
        bb = cv2.boundingRect(approx)
        x, y, w, h = bb
        roi = cv2.rectangle(roi, bb, (0, 0, 255), 5)

    cv2.imshow("roi",cv2.resize( roi, (720, 500)))

  key = cv2.waitKey(1) & 0xFF
  if key == ord("a"):
    print(lo)
    print(hi)
  if key == ord('s'):
    show_boxes = not show_boxes
    print(f"Showboxes: {show_boxes}")
  if 27 == key:
    break
