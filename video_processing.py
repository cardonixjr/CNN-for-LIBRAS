print("Importando pacotes para processamento de videos")
import mediapipe as mp
import os, cv2
from mediapipe_utls import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
print("Pacotes importados com sucesso")
    
def load_npy(actions, data_path):
    print("carregando dados do numpy")
    DATA_PATH = os.path.join(data_path)
    
    label_map = {label:num for num,label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for video_num in range(20):
            video = []
            for frame_num in range(30):
                try:
                    FOLDER_PATH = os.path.join(DATA_PATH, action, str(video_num))
                    res = np.load(os.path.join(FOLDER_PATH,f'{frame_num}.npy'))
                    video.append(res)
                except:
                    continue
            
            sequences.append(np.array(video))
            labels.append(label_map[action])
##            print(f"Loaded video {video_num} from {action}")
            
    return (sequences, labels)

def aply_fastDTW(sequences, labels):
    print("aplicando fastdtw")

    count = 0
    from math import inf
    greater = -inf
    greater_sequence = None
    for sequence in sequences:
        if len(sequence) > greater:
            greater = len(sequence)
            greater_sequence = sequence

    adjusted_sequences = []
    greater_index_map = np.arange(0,greater,1)
    for sequence in sequences:
        smaller_index_map = np.arange(0,len(sequence),1)
        distance, path = fastdtw(greater_index_map, smaller_index_map, dist=euclidean)
        new_sequence = np.zeros([len(path), 258])
        count += 1
        for i in range(len(path)):
            new_sequence[i,:] = sequence[path[i][1], :]
        adjusted_sequences.append(new_sequence)
            
    return np.array(adjusted_sequences), greater

def split_and_shuffle(X, Y, perc = 0.1):
  ''' Esta função embaralha os pares de entradas
      e saídas desejadas, e separa os dados de
      treinamento e validação
  '''
  
  from tensorflow.keras.utils import to_categorical
  Y = to_categorical(Y).astype(int)
  # Total de amostras
  tot = len(X)
  # Emabaralhamento dos índices
  indexes = np.arange(tot)
  np.random.shuffle(indexes)
  # Calculo da quantidade de amostras de
  # treinamento
  n = int((1 - perc)*tot)

  Xt = np.array([X[v] for v in indexes[:n]])
  Yt = np.array([Y[v] for v in indexes[:n]])
  Xv = np.array([X[v] for v in indexes[n:]])
  Yv = np.array([Y[v] for v in indexes[n:]])
  return Xt, Xv, Yt, Yv

if __name__ == "__main__":
    mp_hands = mp.solutions.hands # Hands model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    data_path = "MP_Dataset"
    actions = ["abraco", "amigo", "por favor", "obrigado", "casa", "ajuda",
               "alegria", "professor", "brincar", "bom","ruim", "LIBRAS", "saber",
               "parar", "comecar", "dia", "surdo", "comer", "ola", "feliz"]

    sequences, labels = load_npy(actions, data_path)
    x_train, x_test, y_train, y_test = split_and_shuffle(sequences, labels, perc=0.3)
    
