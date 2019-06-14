# Robocup Rescue Simulation Data Generator
The data generator for train the neural network with Robocup Rescue Simulation (RCRS)

## 1. Software Pre-Requisites

- Git
- OpenJDK Java 8+

## 2. Download project from GitHub and decompress RCRS

```bash

$ git clone https://github.com/swhaKo/RCRS-deep-learning.git
$ unzip RCRS-deep-learning/RCRS.zip
```
## 3. Configuration

In this repository, there is configuration file called "config.txt"

You can modify the number of civilians, the number of initial fire building and the number of data sets.

Also you can modifiy the path where datasets are stored.


### Map data constant
NUM_OF_CIVILIANS: The number of civilians in the simulation map

NUM_OF_FIRES: The number of initial fire buildings

LIMIT_TIME_STEP: The start point of disaster scenario time step to save image dataset.

LIMIT_CIVILIAN_HP: The treshold of civilians' HP point which determine civilian is injured or not


### Train and test data constant
TRAIN_START_MAP_NUM: The map data start number for training

TRAIN_END_MAP_NUM: The map data end number for training

TEST_START_MAP_NUM: The map data start number for testing

TEST_END_MAP_NUM: The map data end number for testing


### Directory data constant
DATASET_DIR: The path of data set directory

TRAIN_GENERATED_MAP_DIR: The path of the source of simulation map data for training

TRAIN_GENERATED_IMAGE_DIR: The path of the screenshot image of simulation map data for training

TEST_GENERATED_MAP_DIR: The path of the source of simulation map data for testing

TEST_GENERATED_IMAGE_DIR: The path of the screenshot image of simulation map data for testing




## 4. Execute
```bash

$ python3 train_data_generator.py [Map Name]
$ python3 test_data_generator.py [Map Name]
```

## 5. Map List

- Kobe
- Joao



