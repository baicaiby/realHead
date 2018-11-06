#!/usr/bin/env python
 
import cv2
import os
import csv
import numpy as np
import math
import time
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
from collections import OrderedDict

#to save the orientation data, change the file name to your folders path
localtime = time.strftime("%Y%m%d_%H_%M_%S",time.localtime())
# filename = '/Users/dansixuan/ear_files/'+localtime + '.csv'
filename = os.path.join(os.path.dirname(__file__))+'/'+localtime + '.csv'
fileObject = open(filename, 'w+')
# fileObject = open("ear_data.csv", "w+")
writer = csv.writer(fileObject)
writer.writerow(["roll", "pitch", "yaw"])
fileObject = open(filename, "a")
writer = csv.writer(fileObject)
cap = cv2.VideoCapture(0)

# construct the argument parser and parse the arguments
# use the shape_predictor_68_face_landmarks.dat in the same folder as a defalt
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
    default=os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
    help="path to facial landmark predictor")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor:shape_predictor_68_face_landmarks.dat
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) : 
    # sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    sy = math.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2])
    assert(isRotationMatrix(R))

    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([[x, y, z]])

# LANDMARKS_DICT = OrderedDict([
#     ("Nose_tip", 30),
#     ("Chin", 8),
#     ("LeftEyeLeft", 36),
#     ("RighEyeRight", 45),
#     ("LeftMouth", 48),
#     ("RightMounth", 54)
# ])

#the index of the points we chose in our predictor
landmark_index = (30,8,36,45,48,54)
image_points = np.zeros(shape=(6,2), dtype="double")
# Read Image
# im = cv2.imread("/Users/dansixuan/headPose.jpg");
# size = im.shape

while(True):
    ret, image = cap.read()
    time.sleep(.01)
    # image = cv2.imread("/Users/dansixuan/headPose.jpg");
    size = image.shape
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)
    pts_order=0

    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        j = 0
        for i in range(len(landmark_index)):
            image_points[j] = shape[landmark_index[i]]
            j = j+1

        # for (name,i) in LANDMARKS_DICT.items():
        #     x,y = shape[i]
        #     pts = (x,y)
        #     image_points[pts_order] = pts
        #     print "pts: ",pts
        #     pts_order = pts_order+1
        #     print "pts_order ", pts_order
         
        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                 
                                ])
         
         
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

        # print "pts",pts,'/n',"image_points: ", image_points
        print "Camera Matrix :\n {0}".format(camera_matrix)
         
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
         
        print "Rotation Vector:\n {0}".format(rotation_vector)
        print "Translation Vector:\n {0}".format(translation_vector)

        # euler
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        angles = rotationMatrixToEulerAngles(rotation_matrix)
        text= 'roll(x): '+ str("%.2f" %angles[0,0]) +', pitch(y): '+ str("%.2f" %angles[0,1])+', yaw(z): '+ str("%.2f" %angles[0,2])
        # print 'angles, anticlockwise +, ', '[roll(x), pitch(y), yaw(z): ]', angles
        writer.writerow([angles[0,0],angles[0,1],angles[0,2]])
        cv2.putText(image,text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255))

         
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
         
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (255,255,0), -1)
         
         
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
         
        cv2.line(image, p1, p2, (0,0,255), 1)


        # Display image
    cv2.imshow("Output", image)
    cv2.waitKey(1) & 0xFF
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()