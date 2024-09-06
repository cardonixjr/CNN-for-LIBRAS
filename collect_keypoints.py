import cv2, os, mediapipe
import numpy as np
from mediapipe_utls import *

def collect_keypoints(actions, data_path):
  '''
  Função captura a tela em n sequências de N frames para geração de um
  database para treinamento e validação dos sinais
  '''
  DATA_PATH = os.path.join(data_path)

  # Actions that i'll try to detect
  no_sequences = 20
  sequence_lenght = 30

  for action in actions:
    for sequence in range(20):
      try:
        os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
      except:
        pass
  
  video_device = 0
  capture = cv2.VideoCapture(video_device)
  with mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
      performing_gesture = False
      stop_gesture_count = 0
      # loop through action
      for action in actions:
        # loop through videos
        for sequence in range(20):
          frame_count = 0
          # loop through frames
          for frame_num in range(sequence_lenght):
            ret,frame = capture.read()
            image, hand_results = mediapipe_detection(frame, hands)
            image, pose_results = mediapipe_detection(frame, pose)

            blank_image = np.zeros(image.shape, np.uint8) + 255
            
            draw_hand_landmarks(blank_image, hand_results)
            draw_pose_landmarks(blank_image, pose_results)

            if frame_num == 0:
              cv2.putText(image, 'STARTING COLLECTION',(120,200),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),4,cv2.LINE_AA)
              cv2.putText(image, 'collectiong frames for {} Video Number {}'.format(action, sequence),(15,12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),4,cv2.LINE_AA)
              cv2.imshow('opencv frame', blank_image)
              cv2.waitKey(1000)

            else:
              if frame_num == 15:
                cv2.imwrite(f"prints/{action}_{sequence}.png",blank_image)
              cv2.putText(image, 'collectiong frames for {} Video Number {}'.format(action, sequence),(15,12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),4,cv2.LINE_AA)
              
            keypoints, handedness = pose_hands_extract_keypoints(hand_results, pose_results)

            if not handedness == ['none','none']:
              stop_gesture_count = 0
              if performing_gesture == False:
                performing_gesture = True
            if handedness == ['none', 'none']: stop_gesture_count += 1

            if stop_gesture_count == 5: performing_gesture = False

            if performing_gesture:
              frame_count += 1
              npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
              np.save(npy_path,keypoints)
              
##            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))

##            npy_path = os.path.join('prints', str(frame_num))
##            cv2.imwrite(npy_path + '.png',image)

            cv2.imshow('opencv frame', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
              break
            
          print(frame_count)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp_hands = mp.solutions.hands # Hands model
    mp_pose = mp.solutions.pose # Pose model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    data_path = "MP_Dataset"

    actions = ["começar", "dia", "surdo", "comer", "ola", "feliz"]

    collect_keypoints(actions, data_path)
  
