from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import mediapipe as mp
from video_processing import *
import matplotlib.pyplot as plt
from CNN import CNN

model_path = os.path.join("models/20_sign_model.h5")
data_path = "MP_Dataset"

model = load_model(model_path)

VOCABULARY_LENGTH = 20

actions = ["abraco", "amigo", "por favor", "obrigado", "casa", "ajuda", "alegria", "professor", "brincar", "bom",
                "ruim", "LIBRAS", "saber", "parar", "comecar", "dia", "surdo", "comer", "ola", "feliz"]

sequences, labels = load_npy(actions, data_path)
adj_sequences, length = aply_fastDTW(sequences, labels)
x_train, x_test, y_train, y_test = split_and_shuffle(adj_sequences, labels, perc=0.3)
x_train = np.reshape(x_train, (-1,30,258,1))
x_test = np.reshape(x_test, (-1,30,258,1))

cnn = CNN(actions, VOCABULARY_LENGTH)
cnn.load_model(model_path=model_path)

en_labels = ["Hug", "Friend", "Please", "Thank you", "House", "Help", "Joy", "Teacher", "to Play", "Good",
            "Bad", "LIBRAS", "to Know", "Stop", "Start", "Day", "Deaf", "to Eat", "Hello", "Happy"]

cnn.evaluate(x_test, y_test,en_labels=en_labels)