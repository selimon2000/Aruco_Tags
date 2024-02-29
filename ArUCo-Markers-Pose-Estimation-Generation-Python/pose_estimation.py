import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

def pose_estimation(frame, matrix_coefficients, distortion_coefficients):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None and len(ids) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            print("rvec\n", rvec)
            print("tvec\n", tvec)

            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            # cv2.aruco.drawDetectedMarkers



            axis_length = 0.1

            # Project 3D points of the axes onto the image plane
            axis_points, _ = cv2.projectPoints(np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]), rvec, tvec, matrix_coefficients, distortion_coefficients)

            # Convert the points to integer coordinates
            axis_points = np.int32(axis_points).reshape(-1, 2)

            # Draw the axis lines on the image
            cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[1]), (0, 0, 255), 2)  # X-axis (Red)
            cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[2]), (0, 255, 0), 2)  # Y-axis (Green)
            cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[3]), (255, 0, 0), 2)  # Z-axis (Blue)

            # Display the image with the axis lines
            cv2.imshow('Estimated Pose with Axis', frame)            

    return frame



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]

    # k = [918.7401733398438, 0.0, 647.2181396484375, 0.0, 918.3084716796875, 345.8296203613281, 0.0, 0.0, 1.0]
    # d = [0.0, 0.0, 0.0, 0.0, 0.0]
    k = np.array([[918.7401733398438, 0.0, 647.2181396484375], [0.0, 918.3084716796875, 345.8296203613281], [0.0, 0.0, 1.0]])
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output = pose_estimation(frame, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()