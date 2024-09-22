from os import listdir
import numpy as np
import pandas as pd
import cv2 as cv
import pytesseract as pyt
from PIL import Image
import re


# Useful colors
RED = (0, 0, 225)
GREEN = (0, 225, 0)
BLUE = (225, 0, 0)

# Key press numbers
ENTER = 13
ESC = 27

# Directory to go through
DIR_NAME = './img/house/1983 - 1983/'


# Function to pass to trackbars
def nop(x):
  pass

for img_name in sorted(listdir(DIR_NAME)):

  # Grab scan and convert to HSV
  img = cv.imread(DIR_NAME + img_name)
  hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

  # Range of HSV values for mask
  lo = np.asarray([11, 19, 64]) if 'house' in DIR_NAME else np.asarray([11, 19, 64])
  hi = np.asarray([105, 66, 255]) if 'house' in DIR_NAME else np.asarray([105, 66, 255])

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

  # Part 2: OCR + corrections
  fields = ['Name', 'Party', 'Constituency', 'Date of Birth', 'Education']
  df = pd.DataFrame(columns=fields)
  for box in text_boxes:
    # Get crop of image and send to Pytesseract
    x, y, w, h = box
    crop = img[y:(y + h), x:(x + w)]
    string = pyt.image_to_string(cv.cvtColor(crop, cv.COLOR_BGR2RGB))

    # Sort OCR result into fields
    filtered_string = filter(lambda s: s, string.split('\n'))
    text_fields = []
    for line in filtered_string:
      if (':' not in line) and all(map(lambda x: x not in line, fields)) and len(text_fields) != 0:
        text_fields[-1] += ' ' + line
      else:
        text_fields.append(line)

    # Extract data from fields
    dictionary = dict()
    for line in text_fields:
      if any(map(lambda x: x in line, fields)):
        # Most fields will include a colon, sometimes there is a misprint
        if ':' in line:
          [key, value] = line.split(':')
          dictionary[key.strip()] = value.strip()
        # Extract political party if listed
        if 'Name' in line:
          m = re.search('\(.+\)', dictionary['Name'])
          if m:
            dictionary['Party'] = m.group()[1:-1]
            dictionary['Name'] = dictionary['Name'][:(m.start()-1)]

    # Show image for corrections
    Image.fromarray(cv.cvtColor(crop, cv.COLOR_BGR2RGB)).show()

    # Correct fields and figure out the rest
    print('Are the following fields correct? If yes, type "y". Otherwise, type the correct string.')
    for field in fields:
      user_input = input(f'{field}:{dictionary[field] if field in dictionary.keys() else ''}\n')
      if user_input == 'y':
        continue
      dictionary[field] = user_input

    # Append dictionary information to DataFrame
    series = pd.Series(dictionary)
    df = pd.concat([df, series.to_frame().T], ignore_index=True)
    print(df)

  # Dump DataFrame into CSV
  df.to_csv(f'temp/csv/{img_name[:-4]}.csv', index=False)      
