##print("INFO importando pacotes")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import mediapipe as mp
import os
from mediapipe_utls import *
from fastdtw import fastdtw
from CNN import CNN

if __name__ == "__main__":
    mp_hands = mp.solutions.hands # Holistic model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    VOCABULARY_LENGTH = 20

    actions = ["abraco", "amigo", "por favor", "obrigado", "casa", "ajuda", "alegria", "professor", "brincar", "bom",
                "ruim", "LIBRAS", "saber", "parar", "comecar", "dia", "surdo", "comer", "ola", "feliz"]

##    actions = ["obrigado", "por favor", "amigo", 'ajuda','gostar', 'professor','brincar', 'livro', 'carinho', 'casa']
    
    cnn = CNN(actions, VOCABULARY_LENGTH)
    cnn.load_model("models/20_sign_model.h5")

    sequence, sentence = [], []
    
    print("iniciando captura por vídeo")
    video_device = 0
    capture = cv2.VideoCapture(video_device)

    last_detections = []
    frame_num = 0
    new_sequence = []
    performing_gesture = False
    stop_gesture_count = 0

    n = 0

    text=""
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while capture.isOpened():
                n+=1
                ret,frame = capture.read()
                image, hand_results = mediapipe_detection(frame, hands)
                image, pose_results = mediapipe_detection(frame, pose)
                draw_hand_landmarks(image, hand_results)
                draw_pose_landmarks(image, pose_results)

                keypoints, handedness = pose_hands_extract_keypoints(hand_results, pose_results)

                if not handedness == ['none','none']:
                    stop_gesture_count = 0
                    if performing_gesture == False:
                        performing_gesture = True
                        new_sequence = []
                        
                if handedness == ['none', 'none']:stop_gesture_count += 1
                if stop_gesture_count == 10:performing_gesture = False
                if performing_gesture and len(new_sequence) < 30:new_sequence.append(keypoints)
                elif performing_gesture == False and new_sequence != []:
                    sequence = np.array(new_sequence)

                    ### aply fastdtw
                    distance, path = fastdtw(np.arange(0,30,1), np.arange(0,len(sequence),1)) #, dist=euclidean)
                    new_fastdtw= np.zeros([len(path), 258])
                    
                    for i in range(len(path)):
                        new_fastdtw[i,:] = sequence[path[i][1], :]
                    sequence = new_fastdtw
                    ### end of fastdtw
                    
                    res = cnn.model.predict(np.reshape(sequence,(-1,30,258,1)), verbose=0)
                    now_sign = np.argmax(res[0])
                    perc = max(res[0])
                    last_detections.append(now_sign)
                    if len(last_detections) > 10: last_detections = last_detections[-30:]
                    
                    text = 'Prediction: %.3g%% %s'%(perc * 100, actions[now_sign])

##                    if not frame_num % 10:
##                        path = os.path.join('prints', "Luciano "+str(frame_num))
##                        print(f"salvando em {path}")
##                        cv2.imwrite(path + ".png",image)
                    new_squence = []

                cv2.rectangle(image, (0,0),(350, 50), (0,0,0),-1)
                cv2.putText(image, text, (25,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(n), (600,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('opencv frame', image)

                frame_num += 1
                if frame_num % 30 == 0:
                    print("print")
                    teste = "cccccc"
                    cv2.imwrite(f"prints/{teste}{frame_num}.png",image)
                    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        capture.release()
        cv2.destroyAllWindows()
