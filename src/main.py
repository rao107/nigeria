import os
import pickle
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

# Folder for saving bounding boxes of text
BOX_FOLDER = 'temp/box'
if not os.path.exists(BOX_FOLDER):
  os.makedirs(BOX_FOLDER)

# Folder for saving CSV spreadsheets
CSV_FOLDER = 'csv'
if not os.path.exists(CSV_FOLDER):
  os.makedirs(CSV_FOLDER)

# Directory to go through
DIR_NAME = './img/house/1999 - 2003/'

# Flag if scan is house or senate
IS_HOUSE = 'house' in DIR_NAME

# Size of image when displayed
SCALED_SIZE = (550, 500)  # (x, y)


# Function to pass to trackbars
def nop(x):
  pass

for img_name in sorted(os.listdir(DIR_NAME)):
  # Ask if want to redo work
  if os.path.exists(f'{CSV_FOLDER}/{img_name[:-4]}.csv'):
    user_input = input(f'Data already extracted for {DIR_NAME+img_name}. Extract again? (y/N)')
    if user_input != 'y':
      continue

  # Ask if already segmented image should be used
  text_boxes = []
  if os.path.exists(f'{BOX_FOLDER}/{img_name[:-4]}.pickle'):
    user_input = input('Bounding boxes have already been created. Should they be used again? (y/N)\n')
    if user_input == 'y':
      try:
        with open(f'{BOX_FOLDER}/{img_name[:-4]}.pickle', 'rb') as f:
          text_boxes = pickle.load(f)
      except Exception as ex:
        print(ex)
  
  # Whether bounding box set up should be done
  loop = not bool(text_boxes)

  # Grab scan
  img = cv.imread(DIR_NAME + img_name)

  if loop:
    # Convert scan to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Range of HSV values for mask
    lo = np.asarray([11, 19, 64]) if IS_HOUSE else np.asarray([8, 26, 44])
    hi = np.asarray([99, 66, 255]) if IS_HOUSE else np.asarray([100, 100, 255])

    # Creating window with trackbars to change `lo` and `hi`
    cv.namedWindow('mask')
    cv.createTrackbar('H_lo', 'mask', lo[0], 255, nop)
    cv.createTrackbar('S_lo', 'mask', lo[1], 255, nop)
    cv.createTrackbar('V_lo', 'mask', lo[2], 255, nop)
    cv.createTrackbar('H_hi', 'mask', hi[0], 255, nop)
    cv.createTrackbar('S_hi', 'mask', hi[1], 255, nop)
    cv.createTrackbar('V_hi', 'mask', hi[2], 255, nop)

  # Part 1: Create bounding boxes
  while loop:
    # Update `lo` and `hi` based on trackbars
    lo[0] = cv.getTrackbarPos('H_lo', 'mask')
    lo[1] = cv.getTrackbarPos('S_lo', 'mask')
    lo[2] = cv.getTrackbarPos('V_lo', 'mask')
    hi[0] = cv.getTrackbarPos('H_hi', 'mask')
    hi[1] = cv.getTrackbarPos('S_hi', 'mask')
    hi[2] = cv.getTrackbarPos('V_hi', 'mask')

    # Create mask in `mask` window
    mask = cv.bitwise_not(cv.inRange(hsv, lo, hi))

    # Find contours using mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
    diff = cv.absdiff(dilate, mask)
    edges = diff
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

    # Use contours to get bounding boxes of text
    big_contours = filter(lambda cnt: 300000 < cv.contourArea(cnt) and cv.contourArea(cnt) < 1000000, contours)
    text_boxes = [cv.boundingRect(cnt) for cnt in big_contours]
    roi = cv.copyTo(img, mask=img)
    for box in text_boxes:
      roi = cv.rectangle(roi, box, RED if IS_HOUSE else GREEN, 5)
    
    # Show mask and roi
    cv.imshow('roi', cv.resize(roi, SCALED_SIZE))
    cv.imshow('mask', cv.resize(mask, SCALED_SIZE))

    # Check for user input
    key = cv.waitKey(1) & 0xFF
    if key == ESC:
      cv.destroyAllWindows()
      break
    elif key == ENTER:
      cv.destroyAllWindows()
      loop = False
  
  # Part 2: Add extra boxes
  extra_boxes = []
  if loop:
    cv.namedWindow('roi')
    prev = (-1, -1)
    scalex, scaley = SCALED_SIZE
    xscaling = len(img[0])/scalex
    yscaling = len(img)/scaley
    def mouseCallback(event, x, y, flags, param):
      global prev
      if event == cv.EVENT_LBUTTONDOWN:
        if prev == (-1, -1):
          prev = (x, y)
        else:
          px, py = prev
          extra_boxes.append((int(px * xscaling), int(py * yscaling), int((x-px) * (len(img[0])/550)), int((y-py) * (len(img)/500))))
          prev = (-1, -1)
    cv.setMouseCallback('roi', mouseCallback)

  while loop:
    roi = cv.copyTo(img, mask=img)
    for box in text_boxes:
      roi = cv.rectangle(roi, box, RED if IS_HOUSE else GREEN, 5)
    for box in extra_boxes:
      roi = cv.rectangle(roi, box, BLUE, 15)
    cv.imshow('roi', cv.resize(roi, SCALED_SIZE))

    # Check for user input
    key = cv.waitKey(1) & 0xFF
    if key == ESC:
      cv.destroyAllWindows()
      break
  text_boxes += extra_boxes
  
  # Interlude: Save bounding boxes of text
  if text_boxes:
    try:
      with open(f'{BOX_FOLDER}/{img_name[:-4]}.pickle', "wb") as f:
        pickle.dump(text_boxes, f)
    except Exception as ex:
      print(ex)

  # Part 3: OCR + corrections
  fields = ['Name', 'Party', 'Constituency', 'Date of Birth', 'Education']
  df = pd.DataFrame(columns=fields)
  # Sort text boxes to add constituency state column easier
  if len(text_boxes) > 8:
    temp_boxes = sorted(text_boxes, key=lambda box: box[0])
    left_boxes = sorted(temp_boxes[:8], key=lambda box: box[1])
    right_boxes = sorted(temp_boxes[8:], key=lambda box: box[1])
    sorted_boxes = left_boxes + right_boxes
  else:
    sorted_boxes = sorted(text_boxes, key=lambda box: box[1])
  for box in sorted_boxes:
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
          key = line.split(':')[0]
          value = line.split(':')[1]
          if key.strip() in fields:
            dictionary[key.strip()] = value.strip()
        # Extract political party if listed
        if 'Name' in line and 'Name' in dictionary.keys():
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

  # Dump DataFrame into CSV
  df.to_csv(f'{CSV_FOLDER}/{img_name[:-4]}.csv', index=False)      
