---
layout: single
classes: wide
title: "Robocup Rescue Simulation Scenario Generator"
permalink: /resources/Scenario_Generator/
read_time: true
toc: false
---
We developed a scenario generator to train the machine running model. In deep learning with image processing, it takes a lot of image data to train the machine learning model. And it takes a lot of work to generate this amount of data by a human. Therefore, we have developed a scenario generator that automatically generates simulated image data. The detail about [The RoboCup Rescue Simulation (RCRS)](https://rescuesim.robocup.org/) and our project, see this [page](https://swhako.github.io/swha/resources/Intro/).
{: style="text-align: justify;"}

Our scenario generator run as follows:
1. Input the setting of scenario (i.e., the number of civilians and fires, location of the rescue team, etc.), the number of scenarios to create and the size of grid to divide into.
2. Write the configuration file and choose whether use [Feature-Hightlight data augmentation](https://swhako.github.io/swha/resources/Augmentation/) or not.
3. As the simulation runs, the generator automatically parses the screenshot image data and text data (i.e., the number of injured civilians, rescue team location, etc.)

## 1. Software Pre-Requisites
- Git
- OpenJDK Java 8+

## 2. Download project from GitHub and decompress RCRS
```bash
$ git clone https://github.com/swhaKo/Scenario_Generator.git
$ cd RCRS-deep-learning/
$ unzip RCRS.zip
```

## 3. Configuration
In this repository, there is configuration file called *config.txt*. You can modify the number of civilians, the number of initial fire building and the number of data sets. Also you can modifiy the path where datasets are stored.  
{: style="text-align: justify;"}

### Constant of Scenario
NUM_OF_CIVILIANS: The number of civilians in the simulation map  
NUM_OF_FIRES: The number of initial fire buildings  
FH_DATA_AUG: Whether to create the scenario using [Feature-Hightlight data augmentation](https://swhako.github.io/swha/resources/Augmentation/)

### Constant of train and test to ML model
TRAIN_START_MAP_NUM: The map data start number for training  
TRAIN_END_MAP_NUM: The map data end number for training  
TEST_START_MAP_NUM: The map data start number for testing  
TEST_END_MAP_NUM: The map data end number for testing  
DATASET_DIR: The path of data set directory to save

## 4. Execute
After write the configuration file, you should generate the disaster scenarios on RCRS
```bash
$ python3 ScenarioGen_train.py [Map Name]
$ python3 ScenarioGen_test.py [Map Name]
```
When you execute the Scenario Generator you can see the automatically run the two terminal and simulator. The simulation viewer that is displayed on the full screen can be closed. One of the terminals run as a server. Server terminal shows the setting of the scenario initially and how much scenario creation is in progress. And the other terminal s a client. Client terminal runs the RCRS and sends simulation data to the server. After finish to run one scenario, it is automatically closed and run the next scenario. This server-client run locally, and it is similar to the official RCRS. If you want to know the detail, please refer to the official RCRS homepage. The screenshot of Scenario Generator is the figure as below:
{: style="text-align: justify;"}

![Robocup Rescue Simulation]({{site.url }}{{site.baseurl }}/assets/images/Screenshot.png){: .align-center}



## 5. Map List
For `[Map Directory Name]`, only scenarios within the `[Scenario Generator Directory]/RCRS/baseMap]` directory are possible. If there are any scenarios you want, they should be moved to that directory. In our program, we provide the following scenario samples:
{: style="text-align: justify;"}

- Kobe
- Joao


## 6. Structure
* Dataset directory
  * raw
    * train
      * generated_image
        * map_name
          * map_name_1
            * Image
            * Image-CiviliansView
            * Label
            * Parse
          * map_name_2
            * Image
            * Image-CiviliansView
            * Label
            * Parse
          * map_name 3
            * Image
            * Image-CiviliansView
            * Label
            * Parse
          * ...
            * ...
      * generated_map
        * map_name
          * map_name_1
            * config
            * map
          * map_name_2
            * config
            * map
          * map_name_3
            * config
            * map
          * ...
            * ...
    * test
      * generated_image
        * ...
      * generated_map
        * ...

In the scenario directory which created by the generator, there is *raw* directory. In the *raw* directory, there are *train* and *test* directories and each directory contains *generated_image* and *generated_map* directories. In each directory, you can see that there are as many scenarios created by the directory in the scenario name. In the *generated_image* and *generated_map* of scenario directory, there are four directories and each directory use as follows:
{: style="text-align: justify;"}

* Image: the simulation of screenshot *png* image files.
* Image-CviliansView: the simulation of screenshot *png* image files which show the location of civilians as green dot.
* Label: the label file of scenario.
* Parse: the parsed data of simulation which related to injured civilians location.


## 7. Download Link
[Robocup Rescue Simulation Scenario Generator GitHub Page](https://github.com/swhaKo/Scenario_Generator)

