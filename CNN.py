import cv2
import numpy as np
import mediapipe as mp
import os
from video_processing import *
import matplotlib.pyplot as plt

class CNN():
    def __init__(self, actions, VOCABULARY_LENGTH):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, Dense, Flatten
        
        self.actions = actions
        self.model = Sequential()
        self.model.add(Conv2D(64, (3,3), activation='relu', input_shape=(30,258,1)))
        self.model.add(Conv2D(128, (3,3), activation='relu', padding = 'same'))
        self.model.add(Conv2D(64, (3,3), activation='relu', padding = 'same'))
        self.model.add(Conv2D(32, (3,3), activation='relu', padding = 'same'))
        self.model.add(Flatten())
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(VOCABULARY_LENGTH,activation='softmax'))

    def load_model(self, model_path):
        self.model.load_weights(model_path)


    def train(self,x_train, y_train,epochs):
        from tensorflow.keras.callbacks import TensorBoard
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x_train,y_train,epochs=epochs)

        plt.plot(history.history['accuracy'], label='accuracy')
##        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

    def evaluate(self,x_test,y_test,en_labels):
        from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, ConfusionMatrixDisplay
        yhat = self.model.predict(x_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        #self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #results = self.model.evaluate(x_test, y_test)
        #precision =results[2]
        #recall = results[3]
        #print(f"Test Precision: {precision}") # Check index based on compile metrics order
        #print(f"Test Recall: {recall}")
        #f1_score = 2 * (precision * recall) / (precision + recall)
        #print(f1_score)

        from sklearn.metrics import classification_report

        print(classification_report(
            ytrue,
            yhat,
            target_names=en_labels,
            digits=4
        ))

        #mcm = multilabel_confusion_matrix(ytrue, yhat)
        cm = confusion_matrix(ytrue, yhat)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=en_labels)
        
        cm_display.plot(cmap="Blues")
        plt.xticks(rotation=90, ha="right")
        plt.tight_layout()
        plt.show()

        ac = accuracy_score(ytrue,yhat)
        print(ac)
        #print(mcm)

if __name__ == "__main__":
    mp_hands = mp.solutions.hands # Hands model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    data_path = "MP_Dataset"

# 20_sign actions
##    actions = ["obrigado", "por favor", "amigo",'A','B','C','1','2','3','ajuda', 'alegria','escola',
##               'gostar', 'professor', 'mãe', 'pai', 'brincar', 'livro', 'carinho', 'casa']

# 10_sign actions
##    actions = ["obrigado", "por favor", "amigo", 'ajuda','gostar', 'professor','brincar', 'livro', 'carinho', 'casa']

# final 20_sign actions
    VOCABULARY_LENGTH = 20

    actions = ["abraco", "amigo", "por favor", "obrigado", "casa", "ajuda", "alegria", "professor", "brincar", "bom",
                "ruim", "LIBRAS", "saber", "parar", "comecar", "dia", "surdo", "comer", "ola", "feliz"]

    sequences, labels = load_npy(actions, data_path)
    adj_sequences, length = aply_fastDTW(sequences, labels)
    x_train, x_test, y_train, y_test = split_and_shuffle(adj_sequences, labels, perc=0.3)

    x_train = np.reshape(x_train, (-1,30,258,1))
    x_test = np.reshape(x_test, (-1,30,258,1))
    
    cnn = CNN(actions, VOCABULARY_LENGTH)

    print(x_train)
    print(type(x_train))
    print(x_train.shape)

    cnn.train(x_train, y_train, epochs = 50)
    cnn.evaluate(x_test, y_test)


    #cnn.model.save('20_sign_model.h5')
