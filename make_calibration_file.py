import numpy as np
import cv2
import glob
import json

INPUT_FILES =  'data_chess/kamera_czu/*.jpg'
# INPUT_FILES =  'data_chess/kamera_czu/img_8.jpg'
JSON_OUT_FILE = "data_json/camera_czu.json"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

GRID = (7, 9)

objp = np.zeros((GRID[0] * GRID[1],3), np.float32)
objp[:,:2] = np.mgrid[0:GRID[0], 0:GRID[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(INPUT_FILES)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)

counter = [0, 0]
for fname in images:
    counter[0] += 1
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, gray = cv2.threshold(gray, 127, 255, None)


    # Find the chess board corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK

    cv2.imshow('img', gray)
    cv2.waitKey(10)

    ret, corners = cv2.findChessboardCorners(gray, (GRID[0], GRID[1]), flags)

    print(fname, ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        counter[1] += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (GRID[0], GRID[1]), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(600)

print("Total pictures: {}, Valid: {}".format(counter[0], counter[1]))

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}

with open(JSON_OUT_FILE, "w") as f:
    json.dump(data, f, indent=4)


