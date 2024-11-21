#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


def background_remover(path,FRAME_HEIGHT, FRAME_WIDTH, color = True,tresh=35):
    frames = []
    output_frames = []
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    
    for fid in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        success, frame = cap.read()
        if not success or frame is None:
            print(f"Nie można odczytać klatki na pozycji: {fid}")
            continue
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
        frames.append(frame)
    
    if len(frames) == 0:
        print("Brak poprawnych klatek do obliczenia mediany.")
        cap.release()
        exit()
    
    # Konwersja na numpy array i sprawdzanie wartości
    frames = np.array(frames)
    if np.isnan(frames).any() or np.isinf(frames).any():
        print("Frames zawierają nieprawidłowe wartości.")
        cap.release()
        exit()
    
    median = np.median(frames, axis=0)
    if np.isnan(median).any() or np.isinf(median).any():
        print("Mediana zawiera nieprawidłowe wartości.")
        cap.release()
        exit()
    
    median = median.astype(np.uint8)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        
        frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
        diff = cv2.absdiff(median, frame)
        _, diff = cv2.threshold(diff, tresh, 255, cv2.THRESH_BINARY)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        if color:
           
            filtered_frame = cv2.bitwise_and(frame, frame, mask=diff)
        else:
            
            filtered_frame = diff
        
        output_frames.append(filtered_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(output_frames)


# In[2]:


FRAME_HEIGHT, FRAME_WIDTH = 300, 300
vid_folder_path = 'videos'
new_folder_path = 'videos_no_bg'
color = True


# In[4]:


if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

for video in os.listdir(vid_folder_path):
    if video.endswith('moving.mp4'):
        continue
    if video.endswith('.mp4'):
        vid_path = os.path.join(vid_folder_path, video)
        frames = background_remover(vid_path,FRAME_HEIGHT, FRAME_WIDTH,color = color)
        
        output = cv2.VideoWriter(os.path.join(new_folder_path, video), cv2.VideoWriter_fourcc(*'mp4v'),
                                fps=30, frameSize = (FRAME_HEIGHT, FRAME_WIDTH), isColor=color)
        for frame in frames:
            output.write(frame)
        
        output.release()
        cv2.destroyAllWindows()


# In[ ]:




