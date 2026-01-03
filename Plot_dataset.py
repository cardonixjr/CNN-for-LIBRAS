import cv2, os, mediapipe
import numpy as np
from mediapipe_utls import *
from mediapipe.framework.formats import landmark_pb2

def pose_numpy_to_landmarks(pose_np):
    lm_list = landmark_pb2.NormalizedLandmarkList()
    for x, y, z, v in pose_np:
        lm = lm_list.landmark.add()
        lm.x = float(x)
        lm.y = float(y)
        lm.z = float(z)
        lm.visibility = float(v)
    return lm_list


def hand_numpy_to_landmarks(hand_np):
    lm_list = landmark_pb2.NormalizedLandmarkList()
    for x, y, z in hand_np:
        lm = lm_list.landmark.add()
        lm.x = float(x)
        lm.y = float(y)
        lm.z = float(z)
    return lm_list


data_path = "MP_Dataset"
actions = ["abraço", "ajuda", "alegria", "amigo", "bom", "brincar",
           "casa", "comecar", "comer", "dia", "feliz", "LIBRAS",
           "obrigado", "ola", "parar", "por favor", "professor",
           "ruim", "saber", "surdo"]

#loop através de cada sinal

for action in actions:
    for video in range(20):
        frame_count = 0
        for frame_num in range(30):
            try:
                frame_count+=1

                image = np.zeros((640, 480, 3), dtype=np.uint8)+255
                
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles

                data = np.load(os.path.join(data_path, action, str(video), f"{frame_num}.npy"), allow_pickle=True)

                pose_raw = data[:132].reshape(33, 4)
                left_hand_raw = data[132:195].reshape(21, 3)
                right_hand_raw = data[195:].reshape(21, 3)

                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose
                mp_hands = mp.solutions.hands

                pose_landmarks = pose_numpy_to_landmarks(pose_raw)
                left_hand_landmarks = hand_numpy_to_landmarks(left_hand_raw)
                right_hand_landmarks = hand_numpy_to_landmarks(right_hand_raw)

                mp_drawing.draw_landmarks(
                    image,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                mp_drawing.draw_landmarks(
                    image,
                    left_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                mp_drawing.draw_landmarks(
                    image,
                    right_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


                cv2.putText(image, f'{action} {video} {frame_count}',(15,12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),4,cv2.LINE_AA)              

                cv2.imshow('f',image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            except FileNotFoundError:
                print(f"Error: {action}/{video}/{frame_num} not found. ")
                continue

cv2.destroyAllWindows()




