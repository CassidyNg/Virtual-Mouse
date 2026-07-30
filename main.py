import cv2 # Manages webcam, processes image/video, and draws output window
import numpy as np 
import mediapipe as mp # Hand detection and landmark point extraction
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import pyautogui # Controls mouse functions
import math

pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1) # Connects to camera

prev_y = 0
dragging = False

while cap.isOpened():
    ret, frame = cap.read()

    # If frame didn't load properly, exit loop
    if not ret: 
        break

    # Flip horizontally (1 = horizontal, 0 = vertical, -1 = both)
    frame = cv2.flip(frame, 1)

    # Screen dimensions
    screenWidth, screenHeight = pyautogui.size()

    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Index Fingertip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Thumb Fingertip (landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Middle Fingertip (landmark 12)
            middle_tip = hand_landmarks.landmark[12]
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

            # Distance between thumb and index
            pinch_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Distance between index and middle
            index_middle_dist = math.hypot(middle_x - index_x, middle_y - index_y)

            # Convert finger X/Y from camera range to screen range
            frame_margin = 50

            screenX = np.interp(index_x, [frame_margin, w - frame_margin], [0, screenWidth])
            screenY = np.interp(index_y, [frame_margin, h - frame_margin], [0, screenHeight])

            # pyautogui.moveTo(screenX, screenY)

            # print(f"Cam: ({cx}, {cy}) --> Screen: ({screenX}, {screenY})")
            # print(f"Index Tip Position -> X: {cx}, Y: {cy}")

            if index_middle_dist < 60:
                cv2.circle(frame, (index_x, index_y), 15, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (middle_x, middle_y), 15, (0, 255, 255), cv2.FILLED)

                if prev_y != 0:
                    delta_y = prev_y - index_y

                    scroll_amount = int(delta_y * 1)
                    pyautogui.scroll(scroll_amount)

                prev_y = index_y
            else:
                prev_y = 0

                pyautogui.moveTo(screenX, screenY)

                if pinch_dist < 40:
                    cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
                    # pyautogui.click()

                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    cv2.circle(frame, (index_x,index_y), 15, (255, 0, 255), cv2.FILLED)

                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
    
    # Display window
    cv2.imshow('Camera', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources & close window
cap.release()
cv2.destroyAllWindows()