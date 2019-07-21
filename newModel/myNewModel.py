from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, Conv3D, Add, SeparableConv2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling3D
from keras.layers import LSTM, ConvLSTM2D, TimeDistributed, InputLayer, Lambda, BatchNormalization
from keras.models import Model
from keras.layers import merge, Input
from keras.regularizers import l2
from keras.activations import relu

from newModel.resnet_v1 import resnet_v1_model
from newModel.resnet_v2 import resnet_v2_model
from newModel.inception_resnet_v2 import InceptionResNetV2_model
from newModel.inception_v3 import InceptionV3_model
from newModel.resnext import ResNext_model
from newModel.mobilenets import MobileNet_model
from newModel.resnet import ResNet_model
from newModel.densenet import DenseNet_121_model
from newModel.densenet import DenseNet_169_model
from newModel.densenet import DenseNet_201_model
from newModel.densenet import DenseNet_264_model
from newModel.xception import Xception_model
from newModel.wide_resnet import WideResNet_model
from keras import backend as K
import inspect
import io
import sys
import numpy as np
import tensorflow as tf



def ResNet_50(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes)
    return model

def ResNet_50_CBAM(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='cbam_block')
    return model

def ResNet_50_SE(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='se_block')
    return model

def ResNet_50_GCBAM(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='gcbam_block')
    return model

def ResNet_50_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], 
classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def ResNet_101(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 6, 23, 3], classes=num_classes)
    return model

def ResNet_101_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 6, 23, 3], 
classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def ResNet_152(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 8, 36, 3], classes=num_classes)
    return model

def ResNet_152_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 8, 36, 3], 
classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


# ResNext-50

def ResNeXt_50(HEIGHT, WIDTH, num_classes):
    model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes)
    return model

def ResNeXt_50_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_50_CBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='cbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_50_SE_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='se_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_50_GCBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='gcbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model



# ResNext-101

def ResNeXt_101(HEIGHT, WIDTH, num_classes):
    model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes)
    return model

def ResNeXt_101_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101_CBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes, attention_module='cbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101_SE_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes, attention_module='se_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101_GCBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes, attention_module='gcbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model




def DenseNet_121(HEIGHT, WIDTH, num_classes):
    model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_121_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def DenseNet_121_CBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_121_SE_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_121_GCBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def DenseNet_169(HEIGHT, WIDTH, num_classes):
    model = DenseNet_169_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_169_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_169_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_201(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_201_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_201_CBAM(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def DenseNet_201_SE(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def DenseNet_201_GCBAM(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def DenseNet_264(HEIGHT, WIDTH, num_classes):
    model = DenseNet_264_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_264_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_264_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def InceptionResNet_v2(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def InceptionResNet_v2_CBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def InceptionResNet_v2_SE(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def InceptionResNet_v2_GCBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def InceptionResNet_v2_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def Inception_v3(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def Inception_v3_CBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def Inception_v3_SE(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def Inception_v3_GCBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def Inception_v3_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model



def MobileNet(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def MobileNet_CBAM(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes,  attention_module='cbam_block')
    return model

def MobileNet_SE(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def MobileNet_GCBAM(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes,  attention_module='gcbam_block')
    return model

def MobileNet_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def Xception(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def Xception_CBAM(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def Xception_SE(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def Xception_GCBAM(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def Xception_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model



def res3d(inputs, weight_decay):
    # Res3D Block 1
    
    conv3d_1 = Conv3D(64, (3, 7, 7), strides=(1, 2, 2), padding='same',
                                   dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                   kernel_regularizer=l2(weight_decay), use_bias=False,
                                   name='Conv3D_1')(inputs)
    conv3d_1 = BatchNormalization(name='BatchNorm_1_0')(conv3d_1)
    conv3d_1 = Activation('relu', name='ReLU_1')(conv3d_1)

    # Res3D Block 2
    conv3d_2a_1 = Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_2a_1')(conv3d_1)
    conv3d_2a_1 = BatchNormalization(name='BatchNorm_2a_1')(conv3d_2a_1)
    conv3d_2a_a = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_2a_a')(conv3d_1)
    conv3d_2a_a = BatchNormalization(name='BatchNorm_2a_a')(conv3d_2a_a)
    conv3d_2a_a = Activation('relu', name='ReLU_2a_a')(conv3d_2a_a)
    conv3d_2a_b = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_2a_b')(conv3d_2a_a)
    conv3d_2a_b = BatchNormalization(name='BatchNorm_2a_b')(conv3d_2a_b)
    conv3d_2a = Add(name='Add_2a')([conv3d_2a_1, conv3d_2a_b])
    conv3d_2a = Activation('relu', name='ReLU_2a')(conv3d_2a)

    conv3d_2b_a = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_2b_a')(conv3d_2a)
    conv3d_2b_a = BatchNormalization(name='BatchNorm_2b_a')(conv3d_2b_a)
    conv3d_2b_a = Activation('relu', name='ReLU_2b_a')(conv3d_2b_a)
    conv3d_2b_b = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_2b_b')(conv3d_2b_a)
    conv3d_2b_b = BatchNormalization(name='BatchNorm_2b_b')(conv3d_2b_b)
    conv3d_2b = Add(name='Add_2b')([conv3d_2a, conv3d_2b_b])
    conv3d_2b = Activation('relu', name='ReLU_2b')(conv3d_2b)

    # Res3D Block 3
    conv3d_3a_1 = Conv3D(128, (1, 1, 1), strides=(2, 2, 2), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_3a_1')(conv3d_2b)
    conv3d_3a_1 = BatchNormalization(name='BatchNorm_3a_1')(conv3d_3a_1)
    conv3d_3a_a = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_3a_a')(conv3d_2b)
    conv3d_3a_a = BatchNormalization(name='BatchNorm_3a_a')(conv3d_3a_a)
    conv3d_3a_a = Activation('relu', name='ReLU_3a_a')(conv3d_3a_a)
    conv3d_3a_b = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_3a_b')(conv3d_3a_a)
    conv3d_3a_b = BatchNormalization(name='BatchNorm_3a_b')(conv3d_3a_b)
    conv3d_3a = Add(name='Add_3a')([conv3d_3a_1, conv3d_3a_b])
    conv3d_3a = Activation('relu', name='ReLU_3a')(conv3d_3a)

    conv3d_3b_a = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_3b_a')(conv3d_3a)
    conv3d_3b_a = BatchNormalization(name='BatchNorm_3b_a')(conv3d_3b_a)
    conv3d_3b_a = Activation('relu', name='ReLU_3b_a')(conv3d_3b_a)
    conv3d_3b_b = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_3b_b')(conv3d_3b_a)
    conv3d_3b_b = BatchNormalization(name='BatchNorm_3b_b')(conv3d_3b_b)
    conv3d_3b = Add(name='Add_3b')([conv3d_3a, conv3d_3b_b])
    conv3d_3b = Activation('relu', name='ReLU_3b')(conv3d_3b)

    # Res3D Block 4
    conv3d_4a_1 = Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_4a_1')(conv3d_3b)
    conv3d_4a_1 = BatchNormalization(name='BatchNorm_4a_1')(conv3d_4a_1)
    conv3d_4a_a = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_4a_a')(conv3d_3b)
    conv3d_4a_a = BatchNormalization(name='BatchNorm_4a_a')(conv3d_4a_a)
    conv3d_4a_a = Activation('relu', name='ReLU_4a_a')(conv3d_4a_a)
    conv3d_4a_b = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_4a_b')(conv3d_4a_a)
    conv3d_4a_b = BatchNormalization(name='BatchNorm_4a_b')(conv3d_4a_b)
    conv3d_4a = Add(name='Add_4a')([conv3d_4a_1, conv3d_4a_b])
    conv3d_4a = Activation('relu', name='ReLU_4a')(conv3d_4a)

    conv3d_4b_a = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_4b_a')(conv3d_4a)
    conv3d_4b_a = BatchNormalization(name='BatchNorm_4b_a')(conv3d_4b_a)
    conv3d_4b_a = Activation('relu', name='ReLU_4b_a')(conv3d_4b_a)
    conv3d_4b_b = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                      dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay), use_bias=False,
                                      name='Conv3D_4b_b')(conv3d_4b_a)
    conv3d_4b_b = BatchNormalization(name='BatchNorm_4b_b')(conv3d_4b_b)
    conv3d_4b = Add(name='Add_4b')([conv3d_4a, conv3d_4b_b])
    conv3d_4b = Activation('relu', name='ReLU_4b')(conv3d_4b)

    return conv3d_4b


def relu6(x):
    return relu(x, max_value=6)


def mobilenet(inputs, weight_decay):
    conv2d_1a = TimeDistributed(SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_1a'))(inputs)
    conv2d_1a = BatchNormalization(name='BatchNorm_Conv2d_1a')(conv2d_1a)
    conv2d_1a = Activation(relu6, name='ReLU_Conv2d_1a')(conv2d_1a)

    conv2d_1b = TimeDistributed(SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_1b'))(conv2d_1a)
    conv2d_1b = BatchNormalization(name='BatchNorm_Conv2d_1b')(conv2d_1b)
    conv2d_1b = Activation(relu6, name='ReLU_Conv2d_1b')(conv2d_1b)
    conv2d_2a = TimeDistributed(SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_2a'))(conv2d_1b)
    conv2d_2a = BatchNormalization(name='BatchNorm_Conv2d_2a')(conv2d_2a)
    conv2d_2a = Activation(relu6, name='ReLU_Conv2d_2a')(conv2d_2a)

    conv2d_2b = TimeDistributed(SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_2b'))(conv2d_2a)
    conv2d_2b = BatchNormalization(name='BatchNorm_Conv2d_2b')(conv2d_2b)
    conv2d_2b = Activation(relu6, name='ReLU_Conv2d_2b')(conv2d_2b)

    conv2d_2c = TimeDistributed(SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_2c'))(conv2d_2b)
    conv2d_2c = BatchNormalization(name='BatchNorm_Conv2d_2c')(conv2d_2c)
    conv2d_2c = Activation(relu6, name='ReLU_Conv2d_2c')(conv2d_2c)

    conv2d_2d = TimeDistributed(SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_2d'))(conv2d_2c)
    conv2d_2d = BatchNormalization(name='BatchNorm_Conv2d_2d')(conv2d_2d)
    conv2d_2d = Activation(relu6, name='ReLU_Conv2d_2d')(conv2d_2d)

    conv2d_2e = TimeDistributed(SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_2e'))(conv2d_2d)
    conv2d_2e = BatchNormalization(name='BatchNorm_Conv2d_2e')(conv2d_2e)
    conv2d_2e = Activation(relu6, name='ReLU_Conv2d_2e')(conv2d_2e)

    conv2d_3a = TimeDistributed(SeparableConv2D(1024, (3, 3), strides=(2, 2), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_3a'))(conv2d_2e)
    conv2d_3a = BatchNormalization(name='BatchNorm_Conv2d_3a')(conv2d_3a)
    conv2d_3a = Activation(relu6, name='ReLU_Conv2d_3a')(conv2d_3a)

    conv2d_3b = TimeDistributed(SeparableConv2D(1024, (3, 3), strides=(2, 2), padding='same',
                                             depthwise_regularizer=l2(weight_decay),
                                             pointwise_regularizer=l2(weight_decay),
                                             name='SeparableConv2D_3b'))(conv2d_3a)
    conv2d_3b = BatchNormalization(name='BatchNorm_Conv2d_3b')(conv2d_3b)
    conv2d_3b = Activation(relu6, name='ReLU_Conv2d_3b')(conv2d_3b)

    return conv2d_3b


def C3D_CLSTM_C2D(HEIGHT, WIDTH, num_classes, seq_len, weight_decay=0.00005):
    inputs = Input(shape=(seq_len, HEIGHT, WIDTH, 3))

    # Res3D Block
    res3d_featmap = res3d(inputs, weight_decay)

    # GatedConvLSTM2D Block
    clstm2d_1 = ConvLSTM2D(256, (3, 3), strides=(1, 1), padding='same',
                                        kernel_initializer='he_normal', recurrent_initializer='he_normal',
                                        kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                                        return_sequences=True, name='gatedclstm2d_1')(res3d_featmap)
    clstm2d_2 = ConvLSTM2D(256, (3, 3), strides=(1, 1), padding='same',
                                        kernel_initializer='he_normal', recurrent_initializer='he_normal',
                                        kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                                        return_sequences=True, name='gatedclstm2d_2')(clstm2d_1)

    # MobileNet
    features = mobilenet(clstm2d_2, weight_decay)
    gpooling = AveragePooling3D(pool_size=(seq_len // 2, 6, 8), strides=(seq_len // 2, 6, 8),
                                             padding='valid', name='Average_Pooling')(features)
    flatten = Flatten(name='Flatten')(gpooling)
    classes = Dense(num_classes, activation='linear', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
    outputs = Activation('relu', name='Output')(classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
