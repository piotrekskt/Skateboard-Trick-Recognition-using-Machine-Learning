# Skateboard Trick Recognition using Machine Learning

This project applies machine learning to recognize skateboard tricks in video sequences. By training and testing on custom video datasets, it evaluates the performance of ConvLSTM and LRCN models in classifying complex skateboarding tricks. The project involves significant preprocessing and data augmentation to optimize the model's ability to identify subtle differences between movment of the tricks.

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

## Models

The project implements two primary models:
1. **ConvLSTM (Convolutional Long Short-Term Memory)** integrates convolutional and LSTM layers to simultaneously capture spatial and temporal features, ideal for identifying trick dynamics.
2. **LRCN (Long-term Recurrent Convolutional Networks)** processes frames sequentially, first extracting spatial features with CNN and then temporal dependencies with LSTM. This model is faster but may struggle with complex sequences.

### Key Results

- **ConvLSTM** achieved an accuracy of **81.88%** on the augmented grayscale dataset, making it the top-performing model.
- **LRCN** reached **71.25% accuracy** under similar conditions but was less effective for complex tricks involving multi-axis rotations.

---

This work is part of my engineering thesis, with a detailed description available in Polish in the file *Skateboard_Trick_Recognition_Project_Description.pdf* (to be added after defense).
