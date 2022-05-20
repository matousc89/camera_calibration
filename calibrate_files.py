import numpy as np
import cv2
import glob
import json
import uuid
import os

def rotate(img, rot1):
    rot0 = True if img.shape[0] > img.shape[1] else False
    if rot0 == rot1:
        return img
    elif rot0:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

INPUT_FILES =  'data_input/*'
OUTPUT_FILES = "data_output"
JSON_FILE = "data_json/camera_czu.json"

# True = portrait, False = landscape
CALIBRATION_ROTATION = False
FINAL_ROTATION = True

files = glob.glob(os.path.join(OUTPUT_FILES, "*"))

for f in files:
    os.remove(f)


with open(JSON_FILE, "r") as f:
    calibration_data = json.loads(f.read())

mtx = np.array(calibration_data["camera_matrix"])
dist = np.array(calibration_data["dist_coeff"])

images = glob.glob(INPUT_FILES)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)


for fname in images:
    img = cv2.imread(fname)
    img = rotate(img, CALIBRATION_ROTATION)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi # crop
    img = img[y:y + h, x:x + w]
    out_name = os.path.join(OUTPUT_FILES, "img_{}.jpg".format(uuid.uuid4()))


    img = rotate(img, FINAL_ROTATION)
    cv2.imwrite(out_name, img)
    cv2.imshow('img', img)
    cv2.waitKey(10)





# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)
#
# mean_error = 0
# for i in np.arange(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     mean_error += error
#
# print("total error: ", mean_error/len(objpoints))