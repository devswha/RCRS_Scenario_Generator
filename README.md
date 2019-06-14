# Robocup Rescue Simulation Data Generator
The data generator for train the neural network with Robocup Rescue Simulation (RCRS)

## 1. Software Pre-Requisites

- Git
- OpenJDK Java 8+

## 2. Download project from GitHub

```bash

$ git clone https://github.com/swhaKo/RCRS-deep-learning.git
```
## 3. Configuration

In this repository, there is configuration file called "config.txt"

You can modify the number of civilians, the number of initial fire building and the number of data sets.

Also you can modifiy the path where datasets are stored.


### Map data constant

NUM_OF_CIVILIANS: The number of civilians in the simulation map

NUM_OF_FIRES: The number of initial fire buildings

SEQUENCE_TIME_STEP: Do not touch

LIMIT_TIME_STEP: The start point of disaster scenario time step to save image dataset.

LIMIT_CIVILIAN_HP: The treshold of civilians' HP point which determine civilian is injured or not


### Train and test data constant

TRAIN_START_MAP_NUM: The map data start number for training

TRAIN_END_MAP_NUM: The map data end number for training

TRAIN_STEP: Do not touch

TEST_START_MAP_NUM: The map data start number for testing

TEST_END_MAP_NUM: The map data end number for testing


### Directory data constant
DATASET_DIR ./dataset2
TRAIN_GENERATED_MAP_DIR ./dataset2/raw/train/generated_map
TRAIN_GENERATED_IMAGE_DIR ./dataset2/raw/train/generated_image
TRAIN_GENERATED_NPY_DIR ./dataset2/raw/train/generated_map
TEST_GENERATED_MAP_DIR ./dataset2/raw/test/generated_map
TEST_GENERATED_IMAGE_DIR ./dataset2/raw/test/generated_image
MODELS_DIR ./model_new
RESULTS_DIR ./result




## 4. Execute
```bash

$ python3 train_data_generator.py [Map Name]
$ python3 test_data_generator.py [Map Name]
```

## 5. Map List

- Kobe
- Joao


