# Deep Learning For Audio With Python
Code for the "[Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)" series on The Sound of AI YouTube channel.

This repository is a comprehensive collection of resources and code for understanding and implementing deep learning models for audio tasks. It serves as a practical guide, starting from the absolute basics (building neurons and backpropagation from scratch), moving to TensorFlow implementation, and culminating in building a complete Music Genre Classification system using various architectures (MLP, CNN, RNN-LSTM).

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Librosa](https://img.shields.io/badge/librosa-0.11.0-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
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

1.  **Course Overview:** _[Video][1yt]_
2.  **AI, Machine Learning and Deep Learning:** _[Video][2yt]_
3.  **Implementing an Artificial Neuron from Scratch:** _[Video][3yt] | [Code][3cd]_
4.  **Vector and Matrix Operations:** _[Video][4yt] | [Code][4cd]_
5.  **Computation in Neural Networks:** _[Video][5yt] | [Code][5cd]_

---

### Part 2: Neural Networks from Scratch

6.  **Implementing a Neural Network from Scratch:** _[Video][6yt] | [Code][6cd]_
7.  **Training a Neural Network (Backprop & Gradient Descent):** _[Video][7yt] | [Code][7cd]_
8.  **Implementing Backpropagation from Scratch:** _[Video][8yt] | [Code][8cd]_

---

### Part 3: TensorFlow & Audio Preprocessing

9.  **Implementing a Neural Network with TensorFlow 2:** _[Video][9yt] | [Code][9cd]_
10. **Understanding Audio Data for Deep Learning:** _[Video][10yt] | [Code][10cd]_
11. **Preprocessing Audio Data (MFCCs/Spectrograms):** _[Video][11yt] | [Code][11cd]_

---

### Part 4: Music Genre Classification Project (MLP)

12. **Preparing the Dataset:** _[Video][12yt] | [Code][12cd]_
13. **Implementing a Neural Network for Classification:** _[Video][13yt] | [Code][13cd]_
14. **Solving Overfitting:** _[Video][14yt] | [Code][14cd]_

---

### Part 5: Advanced Architectures (CNN & RNN-LSTM)

15. **Convolutional Neural Networks (CNN) Explained:** _[Video][15yt]_
16. **Implementing a CNN for Music Genre Classification:** _[Video][16yt] | [Code][16cd]_
17. **Recurrent Neural Networks (RNN) Explained:** _[Video][17yt]_
18. **Long Short Term Memory (LSTM) Explained:** _[Video][18yt]_
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

[2yt]: https://www.youtube.com/watch?v=1LLxZ35ru_g

[3yt]: https://www.youtube.com/watch?v=qxIaW-WvLDU
[3cd]: <3- Implementing an artificial neuron from scratch/neuron.py>

[4yt]: https://www.youtube.com/watch?v=FmD1S5yP_os
[4cd]: <4- Vector and matrix operations/operations.py>

[5yt]: https://www.youtube.com/watch?v=QUCzvlgvk6I
[5cd]: <5-  Computation in neural networks/computation.py>

[6yt]: https://www.youtube.com/watch?v=0oWnheK-gGk
[6cd]: <6- Implementing a neural network from scratch in Python/mlp.py>

[7yt]: https://www.youtube.com/watch?v=ScL18goxsSg
[7cd]: <7- Training a neural network/train.py>

[8yt]: https://www.youtube.com/watch?v=Z97XGNUUx9o
[8cd]: <8- TRAINING A NEURAL NETWORK/mlp.py>

[9yt]: https://www.youtube.com/watch?v=JdXxaZcQer8
[9cd]: <9- How to implement a (simple) neural network with TensorFlow 2/mlp_tf.py>

[10yt]: https://www.youtube.com/watch?v=m3XbqfIij_Y
[10cd]: <10 - Understanding audio data for deep learning/audio_prep.py>

[11yt]: https://www.youtube.com/watch?v=Oa_d-zaUti8
[11cd]: <11- Preprocessing audio data for Deep Learning/preprocess.py>

[12yt]: https://www.youtube.com/watch?v=szyGiObZymo
[12cd]: <12- Music genre classification: Preparing the dataset/prep_dataset.py>

[13yt]: https://www.youtube.com/watch?v=_xcFAiufwd0
[13cd]: <13- Implementing a neural network for music genre classification/classifier.py>

[14yt]: https://www.youtube.com/watch?v=Gf5DO6br0ts
[14cd]: <14-  SOLVING OVERFITTING in neural networks/classifier.py>

[15yt]: https://www.youtube.com/watch?v=t3qWfUYJEYU

[16yt]: https://www.youtube.com/watch?v=dOG-HxpbMSw
[16cd]: <16- How to Implement a CNN for Music Genre Classification/cnn_classifier.py>

[17yt]: https://www.youtube.com/watch?v=DY82Goknf0s

[18yt]: https://www.youtube.com/watch?v=eCvz-kB4yko

[19yt]: https://www.youtube.com/watch?v=4nXI0h2sq2I
[19cd]: <19- How to Implement an RNN-LSTM Network for Music Genre Classification/lstm.py>