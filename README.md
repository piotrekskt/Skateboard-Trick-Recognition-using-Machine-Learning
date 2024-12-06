# Skateboard Trick Recognition using Machine Learning

This project applies machine learning to recognize skateboard tricks in video sequences. By training and testing on custom video datasets, it evaluates the performance of ConvLSTM and LRCN models in classifying complex skateboarding tricks. The project involves significant preprocessing and data augmentation to optimize the model's ability to identify subtle differences between movment of the tricks.
background_remover.py code was used for removing the background, then augument_dataset.py for applying the augmentation and after that skate_model.py for preprocessing the dataset and traning.
## Dataset

The dataset used in this project was entirely created and curated by me. It features five distinct skateboarding tricks (e.g., Ollie, Backside 180, Pop Shuv-it, Kickflip, and 360 Flip), recorded in diverse locations with varied lighting and perspectives to increase sample diversity. Each trick was performed multiple times to capture different angles and nuances in execution, with a minimum of 50 unique video samples per trick, totaling **307 original samples**.

To enhance the dataset further and maximize model robustness, extensive data preprocessing and augmentation were applied:
- **Background removal** to focus solely on the skateboard and rider, minimizing potential distractions in each sequence.
- **Color variations** including grayscale conversions to mkae sure the models are focusing on the movment not the details.
- **Augumentation** such as rotations, zooms, and flips, which generated additional samples for everytrick. 

The final dataset consists of three main versions:
1. **Original (color, with background)** - 307 samples
2. **Augmented, color, background removed** - 1,535 samples
3. **Augmented, grayscale, background removed** - 1,535 samples

This rigorous preparation ensures that the models are tested on a highly diverse set of samples that are prepered for movment recognition.
![Grey-dataset preview](https://github.com/user-attachments/assets/b47ea034-7b59-4023-8305-b4afa044deb2)

![bgr-dataset preview](https://github.com/user-attachments/assets/65197837-bb81-4e45-8fc9-9d8c0a97c642)

## Models

The project implements two primary models:
1. **ConvLSTM (Convolutional Long Short-Term Memory)** integrates convolutional and LSTM layers to simultaneously capture spatial and temporal features, ideal for identifying trick dynamics.


3. **LRCN (Long-term Recurrent Convolutional Networks)** processes frames sequentially, first extracting spatial features with CNN and then temporal dependencies with LSTM. This model is faster but may struggle with complex sequences.


### Training
Best results for LRCN:
![LRCN](https://github.com/user-attachments/assets/68ac2fd2-7e3f-4df2-a8c2-b010e3362a02)
![LRCN_tricks_data_set_osika_bgr_30_150_2024_11_29__02_38_22_confusion_matrix](https://github.com/user-attachments/assets/148fc257-373a-458c-a9a1-94b0c4444d86)

### Key Results

- **LRCN** reached **82.82% accuracy** on the augumented BGR dataset without background making it the top-performing model. SEQ_LEN = 30 , FRAME_SIZE = 150x150 
- **ConvLSTM** achieved an accuracy of **81.88%** on the augmented grayscale dataset, what was the best performance for this model. SEQ_LEN = 50 , FRAME_SIZE = 100x100 


---

This work is part of my engineering thesis, with a detailed description available in Polish in the file *Skateboard_Trick_Recognition_Project_Description.pdf* (to be added after defense).
