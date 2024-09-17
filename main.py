import numpy as np
import cv2 as cv
import pytesseract as pyt


### PLAN ###
# 1. Use OpenCV to grab bounding boxes of text automatically
# 2. Manually select the text boxes it did not detect
# 3. Pass text to Pytesseract and edit mistakes

img = cv.imread("./img/PXL_20230615_184150415.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lo = np.asarray([11, 19, 64])
hi = np.asarray([105, 66, 255])


def nop(x):
  pass

top_left = None
def draw_rect(event, x, y, flags, param):
  if event == cv.EVENT_FLAG_LBUTTON:
    print(x, y)
    if top_left == None:
      top_left = (x, y)
    else:
      # New bounding box
      top_left = None

cv.namedWindow("mask")
cv.createTrackbar("H_lo", "mask", lo[0], 255, nop)
cv.createTrackbar("S_lo", "mask", lo[1], 255, nop)
cv.createTrackbar("V_lo", "mask", lo[2], 255, nop)
cv.createTrackbar("H_hi", "mask", hi[0], 255, nop)
cv.createTrackbar("S_hi", "mask", hi[1], 255, nop)
cv.createTrackbar("V_hi", "mask", hi[2], 255, nop)


loop = True
show_boxes = False
text_boxes = None

# Phase 1: Creating bounding boxes
while loop:
  mask = cv.bitwise_not(cv.inRange(hsv, lo, hi))
  kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
  dilate = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

  # get absolute difference between dilate and thresh
  diff = cv.absdiff(dilate, mask)
  edges = diff

  lo[0] = cv.getTrackbarPos("H_lo", "mask")
  lo[1] = cv.getTrackbarPos("S_lo", "mask")
  lo[2] = cv.getTrackbarPos("V_lo", "mask")
  hi[0] = cv.getTrackbarPos("H_hi", "mask")
  hi[1] = cv.getTrackbarPos("S_hi", "mask")
  hi[2] = cv.getTrackbarPos("V_hi", "mask")
  cv.imshow("mask", cv.resize(mask, (550, 500)))

  if show_boxes:
    roi = cv.copyTo(img, mask=img)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    text_boxes = list(map(lambda box: cv.boundingRect(box), filter(lambda cnt: cv.contourArea(cnt) > 200000, contours)))
    for box in text_boxes:
      roi = cv.rectangle(roi, box, (0, 0, 255), 5)

    cv.imshow("roi", cv.resize(roi, (550, 500)))
    cv.setMouseCallback("roi", draw_rect)

  key = cv.waitKey(1) & 0xFF
  if key == ord('s'):
    show_boxes = not show_boxes
    print(f"Showboxes: {show_boxes}")
  if key == 27: # ESC , 13 == ENTER
    cv.destroyAllWindows()
    loop = False

# Phase 2: Extracting text
imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
for box in text_boxes:
  x, y, w, h = box
  crop = imgrgb[y:y+h, x:x+w]
  print(pyt.image_to_string(crop).replace('\n', ' '))
  while True:
    cv.imshow('crop', cv.resize(cv.cvtColor(crop, cv.COLOR_RGB2BGR), None, fx=0.5, fy=0.5))
    key = cv.waitKey(1) & 0xFF
    if key == 27: # ESC
      break
