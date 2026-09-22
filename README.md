# 🖋️ Handwritten Character Recognition: HandScript AI

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.9%25-2ECC71)
![License](https://img.shields.io/badge/License-MIT-blue)

> A convolutional neural network that recognises handwritten English capital letters (A-Z), with a Streamlit app (**HandScript AI**) for trying it out.

---

## 📖 Overview

This project trains a deep learning model to read single handwritten capital
letters and tells you which letter (A to Z) it thinks an image shows. It has two
parts that work together:

1. **A Jupyter notebook** that loads the data, prepares it, builds and trains a
   convolutional neural network (CNN), and measures how well it does.
2. **A web app called HandScript AI** that loads the trained model and lets you
   upload your own handwriting (or pick a sample letter) and see the prediction,
   how confident the model is, and how it ranks all 26 letters.

The model reaches about **98.9% accuracy** on a held-out test set it never saw
during training.

---

## ✨ Features

**Notebook**
- Loads and explores the dataset with class counts and sample image grids
- Normalises, reshapes, one-hot encodes, and splits the data (stratified)
- Augments the training images (small rotations, zooms, and shifts)
- Builds a three-block CNN with batch normalisation and dropout
- Trains with early stopping, learning-rate reduction, and best-model checkpoints
- Evaluates with a classification report, a confusion matrix, and the letter
  pairs the model confuses most often
- Saves the model, a label map, and a metrics file for the app

**Web app (HandScript AI)**
- Three input modes: draw a letter on a canvas, upload an image, or pick a sample
- Predicts instantly, with no button to press
- Shows the predicted letter, a confidence score, and the next closest guesses
- Displays the 28x28 image the model actually sees
- Plots the model's confidence across all 26 letters
- Clean, responsive layout that works on phones, tablets, and desktops

---

## 📸 Demo

> _Run `streamlit run app/app.py` and replace this line with a screenshot of the live app._

![HandScript AI demo](sample_images/A.png)

---

## 📚 Dataset

**A-Z Handwritten Alphabets**: 372,450 grayscale 28x28 images of handwritten
capital letters, derived from the NIST dataset.

- 🔗 Kaggle: [A-Z Handwritten Alphabets in .csv format](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- **Format:** a single CSV with no header row. Column `0` is the integer label
  (`0=A ... 25=Z`) and the remaining 784 columns are the flattened pixel values
  of a 28x28 image.
- **Size on disk:** about 700 MB, so it is not stored in this repository. Download
  it from Kaggle and place `A_Z Handwritten Data.csv` in the **project root**
  before training.

The dataset is not perfectly balanced (some letters appear far more often than
others), which the notebook shows in a class-distribution chart.

---

## 🔧 Preprocessing & Augmentation

Before training, each image is:

1. **Normalised** from 0-255 pixel values to the 0-1 range.
2. **Reshaped** to `(28, 28, 1)` so it can go into a CNN.
3. Its label is **one-hot encoded** into 26 classes.

The data is then split with stratification (each split keeps the same letter
proportions):

| Split | Share |
|---|---|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

During training only, the images are augmented on the fly to make the model more
robust to messy handwriting:

- Rotation up to 10 degrees
- Zoom up to 10%
- Width and height shifts up to 10%

---

## 🧠 Model Architecture

A CNN with three convolutional blocks, each using batch normalisation and
dropout, followed by a dense classifier head. Batch normalisation helps the
network train faster, and dropout reduces overfitting on the more common letters.

| Stage | Layers | Output |
|---|---|---|
| Input | - | 28 x 28 x 1 |
| Block 1 | Conv2D(32) -> BN -> Conv2D(32) -> BN -> MaxPool -> Dropout(0.25) | 14 x 14 x 32 |
| Block 2 | Conv2D(64) -> BN -> Conv2D(64) -> BN -> MaxPool -> Dropout(0.25) | 7 x 7 x 64 |
| Block 3 | Conv2D(128) -> BN -> MaxPool -> Dropout(0.25) | 3 x 3 x 128 |
| Head | Flatten -> Dense(256) -> BN -> Dropout(0.5) -> Dense(26, softmax) | 26 |

**Total parameters:** about 443,000.

**Optimizer:** Adam (lr = 0.001) - **Loss:** categorical crossentropy -
**Batch size:** 128 - **Epochs:** up to 30.

---

## 🏋️ Training

The model trains with three callbacks that manage the run automatically:

- **EarlyStopping** (patience 5): stops training once the validation loss stops
  improving, and restores the best weights so it does not overfit.
- **ReduceLROnPlateau** (patience 3, factor 0.5): halves the learning rate when
  progress stalls, which helps it settle into a better minimum.
- **ModelCheckpoint**: saves the best model to `models/best_model.keras` based on
  validation accuracy.

The notebook also plots the training and validation accuracy and loss curves so
you can see how the run went.

---

## 📊 Results

| Metric | Score |
|---|---|
| Train accuracy | **98.2%** |
| Validation accuracy | **98.9%** |
| **Test accuracy** | **98.9%** |

The notebook backs these numbers up with a full per-letter classification report
(precision, recall, and F1 for each letter) and a 26x26 confusion matrix.

Most of the mistakes happen on letters that look alike when handwritten, such as
**I / L**, **O / D / Q**, and **U / V**. These are hard to tell apart even for a
person, especially at 28x28 resolution where the small details are lost.

---

## 🖥️ The Web App (HandScript AI)

Once the model is trained, the app gives it a friendly interface. Choose how to
feed it a letter at the top of the input card:

- **Draw:** write a capital letter with your mouse, finger, or stylus.
- **Upload:** drop in a PNG or JPG of a handwritten letter. The app converts it
  to grayscale, crops and centers the letter in a 28x28 frame, and inverts it if
  needed so it matches the white-on-black style the model was trained on.
- **Samples:** tap any letter A to Z to load a real sample from the test set.

The prediction appears right away and shows:

1. The predicted letter, its confidence score, and the next closest guesses.
2. The 28x28 image the model actually sees.
3. A bar chart of the model's confidence across all 26 letters.

The **How it works** and **About** buttons in the header open short explanations
of the pipeline and the model's facts: test accuracy, parameter count, dataset
size, and number of classes.

---

## 🚀 How to Run

### 1. Clone and install requirements

```bash
git clone <your-repo-url>
cd CodeAlpha_Handwritten-Character-Recognition
pip install -r requirements.txt
```

### 2. Download the dataset

Download `A_Z Handwritten Data.csv` from the [Kaggle link above](#-dataset) and
place it in the project root.

### 3. Train the model

Open the notebook and run it from top to bottom. It saves the trained model to
`models/best_model.keras`:

```bash
jupyter notebook notebook/handwritten_character_recognition.ipynb
```

### 4. Launch the web app

```bash
streamlit run app/app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

---

## 📦 requirements.txt

```text
numpy==1.26.4
pandas==2.3.0
matplotlib==3.10.9
seaborn==0.13.2
scikit-learn==1.8.0
opencv-python-headless==4.10.0.84
tensorflow==2.15.0
streamlit==1.58.0
Pillow==10.0.1
```

---

## 🗂️ Project Structure

```
CodeAlpha_Handwritten-Character-Recognition/
├── notebook/
│   └── handwritten_character_recognition.ipynb   # full training pipeline
├── app/
│   └── app.py                                    # HandScript AI Streamlit app
├── models/
│   └── best_model.keras                          # trained model
├── sample_images/                                # A-Z sample letters for testing
├── requirements.txt
└── README.md
```

---

## 🩹 Troubleshooting

- **TensorFlow install fails with a long-path error on Windows:** this happens
  with the Microsoft Store build of Python because its install path is very long.
  Either enable long-path support in Windows, or create a virtual environment at
  a short path (for example `C:\hcrv`) and install there.
- **App says the model is missing:** make sure you have run the notebook so that
  `models/best_model.keras` exists, and start the app from the project root.

---

## 👤 Author

**Adossi Fred William** | CodeAlpha Machine Learning Internship
