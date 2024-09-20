from os import listdir
import numpy as np
import pandas as pd
import cv2 as cv
import pytesseract as pyt
from PIL import Image


# Useful colors
RED = (0, 0, 225)
GREEN = (0, 225, 0)
BLUE = (225, 0, 0)

# Key press numbers
ENTER = 13
ESC = 27

# Directory to go through
IMG_NAME = './img/house/1983 - 1983/190.jpg'


# Function to pass to trackbars
def nop(x):
  pass

# Grab scan and convert to HSV
img = cv.imread(IMG_NAME)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Range of HSV values for mask
lo = np.asarray([11, 19, 64])
hi = np.asarray([105, 66, 255])

# Creating window with trackbars to change `lo` and `hi`
cv.namedWindow('mask')
cv.createTrackbar('H_lo', 'mask', lo[0], 255, nop)
cv.createTrackbar('S_lo', 'mask', lo[1], 255, nop)
cv.createTrackbar('V_lo', 'mask', lo[2], 255, nop)
cv.createTrackbar('H_hi', 'mask', hi[0], 255, nop)
cv.createTrackbar('S_hi', 'mask', hi[1], 255, nop)
cv.createTrackbar('V_hi', 'mask', hi[2], 255, nop)

# Part 1: Create bounding boxes
text_boxes = []
loop = True
while loop:
  # Update `lo` and `hi` based on trackbars
  lo[0] = cv.getTrackbarPos('H_lo', 'mask')
  lo[1] = cv.getTrackbarPos('S_lo', 'mask')
  lo[2] = cv.getTrackbarPos('V_lo', 'mask')
  hi[0] = cv.getTrackbarPos('H_hi', 'mask')
  hi[1] = cv.getTrackbarPos('S_hi', 'mask')
  hi[2] = cv.getTrackbarPos('V_hi', 'mask')

  # Create and show mask in `mask` window
  mask = cv.bitwise_not(cv.inRange(hsv, lo, hi))
  cv.imshow('mask', cv.resize(mask, (550, 500)))

  # Find contours using mask
  kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
  dilate = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
  diff = cv.absdiff(dilate, mask)
  edges = diff
  contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

  # Use contours to get bounding boxes of text
  big_contours = filter(lambda cnt: cv.contourArea(cnt) > 200000, contours)
  text_boxes = [cv.boundingRect(cnt) for cnt in big_contours]
  roi = cv.copyTo(img, mask=img)
  for box in text_boxes:
    roi = cv.rectangle(roi, box, RED, 5)
  cv.imshow('roi', cv.resize(roi, (550, 500)))

  # Check for user input
  key = cv.waitKey(1) & 0xFF
  if key == ESC:
    cv.destroyAllWindows()
    loop = False
  
"""
# Part 2: Add extra bounding boxes
new_boxes = []
def draw_rect():
  prev = (-1, -1)
  def inner(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
      if prev == (-1, -1):
        pass
      else:
        prev = x, y
  return inner
loop = True
while loop:
  roi = cv.copyTo(img, maske=img)

  # Show already created bounding boxes
  for box in text_boxes:
    roi = cv.rectangle(roi, box, RED, 5)

  # Show added bounding boxes
  for box in new_boxes:
    roi = cv.rectangle(roi, box, BLUE, 5)

  cv.imshow('roi', cv.resize(img, (550, 500)))
  cv.setMouseCallback('roi', draw_rect())

  # Check for user input
  key = cv.waitKey(1) & 0xFF
  if key == ESC:
    cv.destroyAllWindows()
    loop = False
"""

# Part 3: OCR + corrections
df = pd.DataFrame(columns=['Name', 'Constituency', 'Date of Birth', 'Education'])
for box in text_boxes:
  # Get crop of image and send to Pytesseract
  x, y, w, h = box
  crop = img[y:(y + h), x:(x + w)]
  string = pyt.image_to_string(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

  # Show image for corrections
  Image.fromarray(cv.cvtColor(crop, cv.COLOR_BGR2RGB)).show()

  # Some text processing
  filtered_string = filter(lambda s: s, string.split('\n'))

  # Line-by-line corrections
  corrected_string = []
  for line in filtered_string:
    correction = input(line + '\n')
    if not correction:
      # string is correct! :D
      corrected_string.append(line)
    elif correction == 'd':
      # string does not exist, delete
      continue
    else:
      # string needs to be corrected
      corrected_string.append(correction)
    
  # Turn corrected string into dictionary
  temp = []
  for s in corrected_string:
    # Some fields take multiple lines
    if ':' in s:
      temp.append(s)
    else:
      temp[-1] += ' ' + s
  dictionary = {x[0] : x[1] for x in [y.split(':') for y in temp] if x[0] != 'Elected'}

  # Append dictionary information to DataFrame
  series = pd.Series(dictionary)
  df = pd.concat([df, series.to_frame().T], ignore_index=True)
  print(df)

# Dump DataFrame into CSV
df.to_csv(f'temp/csv/{IMG_NAME[2:].replace('/', '-')}.csv')      
