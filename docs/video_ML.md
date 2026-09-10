---
layout: default
title: "Video based Machine Learning Model"
permalink: /video_ML/
---

# Video based Machine Learning Model

In predict the density of injured civilian in RCRS using video-based data, we train the machine learning model using simulation screenshot image sequence. We consider that all frames of video clip are used for training. Therefore, we choose nine frames randomly in images sequence of disaster scenario in RCRS to train the machine learning model and predict the density of the 10th frame which is the next frame of 9th of the nine frames. The machine learning model's inputs are simulated images created and filtered from Plug-in1 and extract images feature using CNN. Afterwards, the Long-short memory (LSTM) extract sequence image feature from CNN. And fully-connected layer output the number of injured people vectors in each grid cell. The different from image-based predictions is add LSTM layer to sequence learning on video.

![Robocup Rescue Simulation]({{ site.baseurl }}/assets/images/Video_ML.png)

## 1. Software Pre-Requisites
- TensorFlow 1.12.0
- Keras 2.2.4

## 2. Download project from GitHub and decompress RCRS
```bash
$ git clone https://github.com/devswha/RCRS_ML_Model.git
```

## 2. Configuration
In this repository, there is configuration file called *config_video.txt*. You can modify the number of civilians, the number of initial fire building and the number of data sets. Also you can modifiy the path where datasets are stored.<br>

### Constant of label the data
LIMIT_TIME_STEP: The start point of disaster scenario time step to save image dataset.<br>
LIMIT_CIVILIAN_HP: The threshold of civilians' HP point which determine civilian is injured or not
LABEL_TRAIN_START_MAP_NUM: The map data start number for training to label
LABEL_TRAIN_END_MAP_NUM: The map data end number for training  to label
LABEL_TEST_START_MAP_NUM: The map data start number for testing to label
LABEL_TEST_END_MAP_NUM: The map data end number for testing to label
LABEL_DATASET_DIR: The path of scenario directory to label

### Constant to convert the data
CONVERT_TRAIN_START_MAP_NUM: The map data start number for training to convert
CONVERT_TRAIN_END_MAP_NUM: The map data end number for training to convert
CONVERT_TEST_START_MAP_NUM: The map data start number for testing to convert
CONVERT_TEST_END_MAP_NUM: The map data end number for testing to convert
CONVERT_DATASET_DIR: The path of scenario directory to convert

### Constant to train and test the ML model
MODEL_TRAIN_START_MAP_NUM: The map data start number for training
MODEL_TRAIN_END_MAP_NUM: The map data end number for training
MODEL_TEST_START_MAP_NUM: The map data start number for testing
MODEL_TEST_END_MAP_NUM: The map data end number for testing
MODEL_DATASET_DIR: The path of scenario directory to train and test
MODELS_DIR: The path of machine learning model to save
RESULTS_DIR: The path of result to save

## 3. Label the density of injured civilians to scenario
We use the machine learning model to predict the location of the injured. And predict the exact location of injured civilians requires considerable computational resources and complexity, so that we divide the simulation map into grid and predict the density of the injured people in each grid cell. This can significantly shorten computational resources, complexity and the time required for training. Furthermore, we expected that the accuracy of the prediction of the injured civilians location will also be increased.

## 4. Execute

Label the generated disaster scenarios data
```bash
$ python3 Labeler_train.py [Map Name] [Grid row] [Grid col]
$ python3 Labeler_test.py [Map Name] [Grid row] [Grid col]
```

```bash
$ python3 Converter_train_video.py [Map Name] [Grid row] [Grid col] [predict_frame]
$ python3 Converter_test_video.py [Map Name] [Grid row] [Grid col] [predict_frame]
```

Train and evaluate the video model using the scripts in the current repository:

```bash
$ python3 Model_train_video.py [Map Name] [Grid row] [Grid col] [predict_frame]
$ python3 Model_test_video.py [Map Name] [Grid row] [Grid col] [predict_frame]
```

## 6. Download Link
[Video-based Machine Learning Model with RCRS GitHub Page](https://github.com/devswha/RCRS_ML_Model)
