from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, Conv3D, Add, SeparableConv2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling3D
from keras.layers import LSTM, ConvLSTM2D, TimeDistributed, InputLayer
from keras.models import Model
from keras.layers import merge, Input

from resnet_v1 import resnet_v1_model

import inspect
import io
import sys
import numpy as np
import tensorflow as tf


def ResNet_v1_56(HEIGHT, WIDTH, num_classes):
    model = resnet_v1_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=56, num_classes=num_classes)
    return model

