
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import sys
from moviepy.editor import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.saving import save_model


num1 = int(sys.argv[1])
num2 = int(sys.argv[2])
num3 = int(sys.argv[3])


seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

FRAME_HEIGHT, FRAME_WIDTH = num1,num1
SEQ_LEN = num2;
if(num3 == 0):
    DATASET_DIR = 'tricks_data_set_osika'
    CH = 3
elif(num3 == 1):
    DATASET_DIR = 'tricks_data_set_osika_grey'
    CH = 1
elif(num3 == 2):
    DATASET_DIR = 'tricks_data_set_osika_bgr'
    CH = 3

CLASSES_LIST = os.listdir(f'{DATASET_DIR}')


print(SEQ_LEN)
print(DATASET_DIR)
print(CLASSES_LIST)

def extract_frames(path_to_video):
    frames_list = []
    video_read = cv2.VideoCapture(path_to_video)
    video_frame_count = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(int(video_frame_count/SEQ_LEN),1)

    for num_frame in range(SEQ_LEN):
        video_read.set(cv2.CAP_PROP_POS_FRAMES,  num_frame * frame_interval)
        success,frame = video_read.read()
        if not success:
            break
        #resize and normalize
        resized_frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))

        #value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        if(num3 == 1):
            normalized_frame = normalized_frame[..., 0]  #greyscale trick
            
        frames_list.append(normalized_frame)

    while len(frames_list) < SEQ_LEN:
      frames_list.append(frames_list[-1])

    video_read.release()
    return frames_list

def preprocess_dataset():
    labels =[]
    video_files_paths = []
    features = []

    for class_idx,class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')

        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        for file_name in files_list:

            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = extract_frames(video_file_path)

            if len(frames) == SEQ_LEN:

                features.append(frames)
                labels.append(class_idx)
                video_files_paths.append(video_file_path)

    features = np.asarray(features)
    labels = np.array(labels)


    return features, labels, video_files_paths



features, labels, video_files_paths = preprocess_dataset()

one_hot_encoded_labels = to_categorical(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.20, shuffle = True, random_state = seed_constant)

print(f"Frame features in train set: {features_train.shape}")
print(f"Frame features in test set: {features_test.shape}")

def create_convlstm_model():
   
    model = Sequential()


    model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = (SEQ_LEN,
                                                                                      FRAME_HEIGHT, FRAME_WIDTH, CH)))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    

    model.add(Flatten())

    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))


    model.summary()

    return model

convlstm_model = create_convlstm_model()

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

convlstm_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

convlstm_model_training_history = convlstm_model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4,shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])

model_evaluation_history = convlstm_model.evaluate(features_test, labels_test)
.
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

model_file_name = f'./CONVlstm_{current_date_time_string}_{DATASET_DIR}_{SEQ_LEN}_{FRAME_HEIGHT}_Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'
save_model(convlstm_model, model_file_name)

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name, title):

    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    epochs = range(len(metric_value_1))
    plt.figure()

    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

    plt.legend([metric_name_1,metric_name_2])

    plt.title(str(plot_name))
    plt.xlabel('epochs')
    plt.ylabel('value')

    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    plt.savefig(f'./{title}_{DATASET_DIR}_{SEQ_LEN}_{FRAME_HEIGHT}_{current_date_time_string}_{plot_name}.png')

def plot_confusion_matrix(model, X_test, y_test, title):
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    
    y_true = np.argmax(y_test, axis=1)

   
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    plt.savefig(f'./{title}_{DATASET_DIR}_{SEQ_LEN}_{FRAME_HEIGHT}_{current_date_time_string}_confusion_matrix.png')
    plt.show()

plot_metric(convlstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss', 'ConvLSTM')

plot_confusion_matrix(convlstm_model,features_test,labels_test,'ConvLSTM')



plot_metric(convlstm_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy','ConvLSTM')

def create_LRCN_model():

    model = Sequential()


    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQ_LEN, FRAME_HEIGHT, FRAME_WIDTH, CH)))

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))


    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(32))

    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

    model.summary()

    return model

LRCN_model = create_LRCN_model();

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

LRCN_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

LRCN_model_training_history = LRCN_model.fit(x = features_train, y = labels_train, epochs = 70, batch_size = 4 ,
                                             shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])

model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)

model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

model_file_name = f'./LRCN_{DATASET_DIR}_{SEQ_LEN}_{FRAME_HEIGHT}_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'

LRCN_model.save(model_file_name)

plot_metric(LRCN_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss','LRCN')

plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy','LRCN')

plot_confusion_matrix(LRCN_model,features_test,labels_test,'LRCN')







