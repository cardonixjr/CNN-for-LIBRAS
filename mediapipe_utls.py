import mediapipe as mp
import numpy as np
import cv2
import time

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def draw_landmarks(image,results):
##    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
##                              mp_drawing.DrawingSpec(color=(255,70,70),thickness=1,circle_radius=1),
##                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

def pose_hands_extract_keypoints(hand_results, pose_results):
    result = []
    result.append(np.array([[res.x,res.y,res.z,res.visibility] for res in pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33*4))
##    print(result)
    handedness = []
    for a in range(2):
        try:
            handedness.append(hand_results.multi_handedness[a].classification[0].label)
        except:
            handedness.append('none')

    if 'Left' in handedness:
##        print('left', end = '//')
        result.append(np.array([[res.x,res.y,res.z] for res in hand_results.multi_hand_landmarks[handedness.index('Left')].landmark]).flatten())
    else:
        result.append(np.zeros(21*3))

    if 'Right' in handedness:
##        print('right')
        result.append(np.array([[res.x,res.y,res.z] for res in hand_results.multi_hand_landmarks[handedness.index('Right')].landmark]).flatten())
    else:
        result.append(np.zeros(21*3))
            
    return (np.concatenate(result), handedness)

def hands_extract_keypoints(hand_results):
    result = []
    handedness = []
    for a in range(2):
        try:
            handedness.append(hand_results.multi_handedness[a].classification[0].label)
        except:
            handedness.append('none')

    if 'Left' in handedness:
##        print('left', end = '//')
        result.append(np.array([[res.x,res.y,res.z] for res in hand_results.multi_hand_landmarks[handedness.index('Left')].landmark]).flatten())
    else:
        result.append(np.zeros(21*3))

    if 'Right' in handedness:
##        print('right')
        result.append(np.array([[res.x,res.y,res.z] for res in hand_results.multi_hand_landmarks[handedness.index('Right')].landmark]).flatten())
    else:
        result.append(np.zeros(21*3))

    return (np.concatenate(result), handedness)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_pose_landmarks(image,results):
    mp_pose = mp.solutions.pose # Pose model
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (50,50,255), thickness=4, circle_radius=3),
        mp_drawing.DrawingSpec(color = (100,100,100)))
        

def draw_hand_landmarks(image,results):
    mp_hands = mp.solutions.hands # Pose model
    mp_drawing_styles = mp.solutions.drawing_styles
    try:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    except:
        pass

if __name__ == "__main__":
    mp_hands = mp.solutions.hands # Hands model
    mp_pose = mp.solutions.pose # Pose model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    video_device = 0
    capture = cv2.VideoCapture(video_device)
    init_time = time.time()
    print_num = 0
    # Cria o modelo para detecção de mãos
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Cria o modelo para detecção da pose
        with mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while capture.isOpened():
                try:
                    ret,frame = capture.read()
                    
                    # Inverte a imagem
                    frame = cv2.flip(frame, 1)
                    
                    frame, hand_results = mediapipe_detection(frame, hands)
                    frame, pose_results = mediapipe_detection(frame, pose)

                    # Desenha os pontos encontrados    
                    draw_hand_landmarks(frame, hand_results)
                    draw_pose_landmarks(frame, pose_results)

                    # Extrai e organiza os pontos
                    keypoints, handedness = hands_extract_keypoints(hand_results)
##                    print(keypoints)

                    # Salva alguns frames de exemplo a cada 10 segundos
##                    if time.time() - init_time >= 1:
##                        init_time = time.time()
##                        print_num += 1
##                        cv2.imwrite(f"prints/utls3{print_num}.png", frame)
                    
                    # Mostra a imagem na tela
                    cv2.imshow('f',frame)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                except:
                    capture.release()
                    cv2.destroyAllWindows()
                    break
    capture.release()
    cv2.destroyAllWindows()
      
