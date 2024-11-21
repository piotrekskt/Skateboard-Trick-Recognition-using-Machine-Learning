#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


input_folder = 'videos_no_bg_bgr'
output_folder = 'aug_videos_no_bg_bgr'
img_size = 300
fps = 30


# In[7]:


def augment_video(input_folder, output_folder,value=1, rotate=False,tag ='none'):
    transformations = ["zoom_in","zoom_out","rotate_15","rotate_neg_15","standard"]
            
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video in os.listdir(input_folder):
        vid_path = os.path.join(input_folder,video)
        cap = cv2.VideoCapture(vid_path)
        frames =[]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = transform(frame,value,rotate)
            frames.append(frame)
        cap.release()
        output_path = os.path.join(output_folder, f"{video[:-4]}_{tag}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_size, img_size))
        for out_frame in frames:
            out.write(out_frame)

        out.release()
            
def transform(frame, value, rotate):
    if not rotate:
        if value > 1:
            coord = None
            angle = 0
            cy, cx = [ i/2 for i in frame.shape[:-1] ] if coord is None else coord[::-1]
                
            rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, value)
            result = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        else:
            w=frame.shape[0]
            h=frame.shape[1]
            new_w = int(w*value)
            new_h = int(h*value)
            resized_frame = cv2.resize(frame,(new_w,new_h))
            result = cv2.copyMakeBorder(resized_frame, (w-new_w)//2, (w-new_w)//2, (h-new_h)//2, (h-new_h)//2, cv2.BORDER_CONSTANT, value=0)
    else:
        center=tuple(np.array(frame.shape[0:2])/2)
        rot_mat = cv2.getRotationMatrix2D(center,value,1.0)
        result =  cv2.warpAffine(frame, rot_mat, frame.shape[0:2],flags=cv2.INTER_LINEAR)
    return result
    

    


# In[8]:


augment_video(input_folder, output_folder,1.1, False,'zoom_in')
augment_video(input_folder, output_folder,0.74, False,'zoom_out')
augment_video(input_folder, output_folder,15, True,'rotate_15')
augment_video(input_folder, output_folder,-15, True,'rotate_neg_15')
augment_video(input_folder, output_folder)

