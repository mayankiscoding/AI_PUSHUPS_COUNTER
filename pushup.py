import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawSpecific = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

pushUpStart = 0
pushUpCount = 0
BG_COLOR = (192, 192, 192) 
cap = cv2.VideoCapture(0)
cv2.namedWindow("MediaPipe Pose", cv2.WINDOW_NORMAL)

while cap.isOpened():
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            break

        image_height, image_width, _ = image.shape

        results = pose.process(image)


        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks is not None:
            nosePoint = (int(results.pose_landmarks.landmark[0].x * image_width), int(results.pose_landmarks.landmark[0].y * image_height))
            leftWrist = (int(results.pose_landmarks.landmark[15].x * image_width), int(results.pose_landmarks.landmark[15].y * image_height))
            rightWrist = (int(results.pose_landmarks.landmark[16].x * image_width), int(results.pose_landmarks.landmark[16].y * image_height))
            leftShoulder = (int(results.pose_landmarks.landmark[11].x * image_width), int(results.pose_landmarks.landmark[11].y * image_height))
            rightShoulder = (int(results.pose_landmarks.landmark[12].x * image_width), int(results.pose_landmarks.landmark[12].y * image_height))

            if distanceCalculate(rightShoulder, rightWrist) < 130:
                pushUpStart = 1
            elif pushUpStart and distanceCalculate(rightShoulder, rightWrist) > 250:
                pushUpCount = pushUpCount + 1
                pushUpStart = 0

            print(pushUpCount)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 70)
            fontScale = 2
            color = (255, 0, 0)
            thickness = 3

            image = cv2.putText(image, "Push-up count:  " + str(pushUpCount), org, font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(1) == ord('q'):
                break

        else:
            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(1) == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()