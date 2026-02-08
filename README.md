# Deep Learning For Audio With Python
Code for the "[Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)" series on The Sound of AI YouTube channel.

This repository is a comprehensive collection of resources and code for understanding and implementing deep learning models for audio tasks. It serves as a practical guide, starting from the absolute basics (building neurons and backpropagation from scratch), moving to TensorFlow implementation, and culminating in building a complete Music Genre Classification system using various architectures (MLP, CNN, RNN-LSTM).

![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![librosa](https://img.shields.io/badge/librosa-9418A8?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-004a96?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

### Note on Versioning
> While this v2 release is fully functional and optimized for current environments, it may differ from the original version shown in the course. The codebase has been updated to reflect modern best practices (e.g. TensorFlow 2.16+, Librosa 0.11+) and improved dependency management. Consequently, the original course version has been deprecated; however, it remains available in the [legacy branch](https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/legacy) for those wishing to follow the video content exactly.

# Table of Contents
* [Dataset Setup (GTZAN)](#dataset-setup-gtzan)
* [Course Structure](#course-structure)
    * [1. Fundamentals & Math](#part-1-fundamentals--math)
    * [2. Neural Networks from Scratch](#part-2-neural-networks-from-scratch)
    * [3. TensorFlow & Audio Preprocessing](#part-3-tensorflow--audio-preprocessing)
    * [4. Music Genre Classification Project](#part-4-music-genre-classification-project-mlp)
    * [5. Advanced Architectures (CNN & RNN)](#part-5-advanced-architectures-cnn--rnn-lstm)
* [How to Run the Scripts](#how-to-run-the-scripts)

---

## Dataset Setup _(GTZAN)_

To run the music genre classification lessons (Part 4 & 5), you will need the GTZAN dataset. We provide an **automated downloader** to handle the acquisition, extraction, and folder organization for you.

* **Quick Start:** Run `python dataset_downloader.py` from the root directory.
* **Prerequisites:** Install requirements.txt.

> **Full Instructions:** Please check the [Instructions GTZAN](Instructions_GTZAN.md) file for detailed help using the downloader script or manual download steps.

---

## Course Structure

### Part 1: Fundamentals & Math

1.  **Course Overview:** _[Video][1yt] | [Slides][1sl]_
2.  **AI, Machine Learning and Deep Learning:** _[Video][2yt] | [Slides][2sl]_
3.  **Implementing an Artificial Neuron from Scratch:** _[Video][3yt] | [Slides][3sl] | [Code][3cd]_
4.  **Vector and Matrix Operations:** _[Video][4yt] | [Slides][4sl]_
5.  **Computation in Neural Networks:** _[Video][5yt] | [Slides][5sl]_

---

### Part 2: Neural Networks from Scratch

6.  **Implementing a Neural Network from Scratch:** _[Video][6yt] | [Code][6cd]_
7.  **Training a Neural Network (Backprop & Gradient Descent):** _[Video][7yt] | [Slides][7sl]_
8.  **Implementing Backpropagation from Scratch:** _[Video][8yt] | [Code][8cd]_

---

### Part 3: TensorFlow & Audio Preprocessing

9.  **Implementing a Neural Network with TensorFlow 2:** _[Video][9yt] | [Code][9cd]_
10. **Understanding Audio Data for Deep Learning:** _[Video][10yt] | [Slides][10sl]_
11. **Preprocessing Audio Data (MFCCs/Spectrograms):** _[Video][11yt] | [Code][11cd]_

---

### Part 4: Music Genre Classification Project (MLP)

12. **Preparing the Dataset:** _[Video][12yt] | [Code][12cd]_
13. **Implementing a Neural Network for Classification:** _[Video][13yt] | [Slides][13sl] | [Code][13cd]_
14. **Solving Overfitting:** _[Video][14yt] | [Slides][14sl] | [Code][14cd]_

---

### Part 5: Advanced Architectures (CNN & RNN-LSTM)

15. **Convolutional Neural Networks (CNN) Explained:** _[Video][15yt] | [Slides][15sl]_
16. **Implementing a CNN for Music Genre Classification:** _[Video][16yt] | [Code][16cd]_
17. **Recurrent Neural Networks (RNN) Explained:** _[Video][17yt] | [Slides][17sl]_
18. **Long Short Term Memory (LSTM) Explained:** _[Video][18yt] | [Slides][18sl]_
19. **Implementing an RNN-LSTM for Music Genre Classification:** _[Video][19yt] | [Code][19cd]_

---

## How to Run the Scripts
To ensure the models and scripts execute correctly, please follow these steps from your terminal:

### 2. Prepare the Environment (Recommended)
Before running inference, ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Navigate to the Lesson Folder
Each class is self-contained. Move into the specific directory for the lesson you are studying:
```bash
cd class/folder/name  # Replace with the specific class directory
```

### 3. Execute the Script
Run the main script using Python:
```bash
python mlp.py  # Replace with the specific script name
```

<!-- Reference links for every chapter:
YouTube videos (#yt), PDF-file slides (#sl) and Jupyter Notebooks (#nb) -->
[1yt]: https://www.youtube.com/watch?v=fMqL5vckiU0
[1sl]: <01 - Course overview/slides/Course overview.pdf>

[2yt]: https://www.youtube.com/watch?v=1LLxZ35ru_g
[2sl]: <02 - Ai, machine learning and deep learning/slides/AI, machine learning and deep learning.pdf>

[3yt]: https://www.youtube.com/watch?v=qxIaW-WvLDU
[3cd]: <03 - Implementing an artificial neuron from scratch/code/artificialneuron.py>
[3sl]: <03 - Implementing an artificial neuron from scratch/slides/Implementing an artificial neuron from scratch.pdf>

[4yt]: https://www.youtube.com/watch?v=FmD1S5yP_os
[4sl]: <04 - Vector and matrix operations/slides/Vector and matrix operations.pdf>

[5yt]: https://www.youtube.com/watch?v=QUCzvlgvk6I
[5sl]: <05 - Computation in neural networks/slides/Computation in neural networks.pdf>

[6yt]: https://www.youtube.com/watch?v=0oWnheK-gGk
[6cd]: <06 - Implementing a neural network from scratch/code/mlp.py>

[7yt]: https://www.youtube.com/watch?v=ScL18goxsSg
[7sl]: <07 - Bagkpropagation and gradient descent/slides/Training a neural network_ Backward propagation and gradient descent.pdf>

[8yt]: https://www.youtube.com/watch?v=Z97XGNUUx9o
[8cd]: <08 - Training a neural network - Implementing back propagation from scratch/code/mlp.py>

[9yt]: https://www.youtube.com/watch?v=JdXxaZcQer8
[9cd]: <09 - How to imlement a simple neural network with TensorFlow/code/mlp.py>

[10yt]: https://www.youtube.com/watch?v=m3XbqfIij_Y
[10sl]: <10 - Understanding audio data for deep learning/slides/Understanding audio data for  deep learning.pdf>

[11yt]: https://www.youtube.com/watch?v=Oa_d-zaUti8
[11cd]: <11 - Preprocessing audio data for deep learning/code/audio_prep.py>

[12yt]: https://www.youtube.com/watch?v=szyGiObZymo
[12cd]: <12 - Music genre classification - Preparing the dataset/code/extract_data_fast.py>

[13yt]: https://www.youtube.com/watch?v=_xcFAiufwd0
[13cd]: <13 - Implementing a neural network for music genre classification/code/mlp_genre_classifier.py>
[13sl]: <13 - Implementing a neural network for music genre classification/slides/Implementing a neural network for music genre calssification.pdf>

[14yt]: https://www.youtube.com/watch?v=Gf5DO6br0ts
[14cd]: <14 - Solving overfitting in neural networks/code/solving_overfitting.py>
[14sl]: <14 - Solving overfitting in neural networks/slides/Solving overfitting in neural networks.pdf>

[15yt]: https://www.youtube.com/watch?v=t3qWfUYJEYU
[15sl]: <15 - How does a convolutional neural network work/slides/How does a convolutional  neural network work.pdf>

[16yt]: https://www.youtube.com/watch?v=dOG-HxpbMSw
[16cd]: <16 - How to implement a CNN for music genre classification/code/cnn_genre_classifier.py>

[17yt]: https://www.youtube.com/watch?v=DY82Goknf0s
[17sl]: <17 - Recurrent Neural Networks explained easily/slides/Recurrent Neural Networks explained easily.pdf>

[18yt]: https://www.youtube.com/watch?v=eCvz-kB4yko
[18sl]: <18 - LSTM networks explained easily/slides/LSTM networks  explained easily.pdf>

[19yt]: https://www.youtube.com/watch?v=4nXI0h2sq2I
[19cd]: <19 - How to implement an RNN-LSTM for music genre classification/code/lstm_genre_classifier.py>