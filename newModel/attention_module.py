from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  elif attention_module == 'gcbam_block':  # GCBAM_block
        net = gcbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def gcbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention_custom(cbam_feature, ratio)
    cbam_feature = spatial_attention_custom(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2

    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])





def spatial_attention_custom(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    GRID_ROW = 4
    GRID_COL = 4

    cellSize_x = K.int_shape(cbam_feature)[1] // GRID_ROW
    cellSize_y = K.int_shape(cbam_feature)[2] // GRID_COL

    custom_features = [[0]*GRID_COL for i in range(GRID_ROW)]
    custom_avg_pool_ary = [[0] * GRID_COL for i in range(GRID_ROW)]
    custom_max_pool_ary = [[0] * GRID_COL for i in range(GRID_ROW)]

    for row in range(0, GRID_ROW):
        for col in range(0, GRID_COL):
            if col == (GRID_COL-1):
                if row == (GRID_ROW-1):
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:, col * cellSize_y:, 0:])(cbam_feature)
                else:
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:(row + 1) * cellSize_x, col * cellSize_y:, 0:])(cbam_feature)
            else:
                if row == (GRID_ROW - 1):
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:, col * cellSize_y:(col + 1) * cellSize_y, 0:])(cbam_feature)
                else:
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:(row+1) * cellSize_x, col * cellSize_y:(col+1) * cellSize_y, 0:])(cbam_feature)

            custom_avg_pool_ary[row][col] = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(custom_features[row][col])
            custom_max_pool_ary[row][col] = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(custom_features[row][col])
            custom_concat = Concatenate(axis=3)([custom_avg_pool_ary[row][col], custom_max_pool_ary[row][col]])
            custom_features[row][col] = Conv2D(filters=1,
                                               kernel_size=kernel_size,
                                               strides=1,
                                               padding='same',
                                               activation='sigmoid',
                                               kernel_initializer='he_normal',
                                               use_bias=False)(custom_concat)

    temp_concat = []
    for row in range(0, GRID_ROW):
        temp_concat.append(Concatenate(axis=2)(custom_features[row]))
    cbam_feature = Concatenate(axis=1)(temp_concat)


    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    # output = Lambda(lambda x: tf.multiply(x[0], x[1]))([input_feature, cbam_feature])
    # return output
    return multiply([input_feature, cbam_feature])


def spatial_attention_custom_2(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    GRID_ROW = 4
    GRID_COL = 4

    cellSize_x = K.int_shape(cbam_feature)[1] // GRID_ROW
    cellSize_y = K.int_shape(cbam_feature)[2] // GRID_COL

    custom_features = [[0]*GRID_COL for i in range(GRID_ROW)]
    custom_avg_pool_ary = [[0] * GRID_COL for i in range(GRID_ROW)]
    custom_max_pool_ary = [[0] * GRID_COL for i in range(GRID_ROW)]

    for row in range(0, GRID_ROW):
        for col in range(0, GRID_COL):
            if col == (GRID_COL-1):
                if row == (GRID_ROW-1):
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:, col * cellSize_y:, 0:])(cbam_feature)
                else:
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:(row + 1) * cellSize_x, col * cellSize_y:, 0:])(cbam_feature)
            else:
                if row == (GRID_ROW - 1):
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:, col * cellSize_y:(col + 1) * cellSize_y, 0:])(cbam_feature)
                else:
                    custom_features[row][col] = Lambda(lambda x: x[0:, row * cellSize_x:(row+1) * cellSize_x, col * cellSize_y:(col+1) * cellSize_y, 0:])(cbam_feature)

            custom_avg_pool_ary[row][col] = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(custom_features[row][col])
            custom_max_pool_ary[row][col] = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(custom_features[row][col])

    temp_concat = []
    for row in range(0, GRID_ROW):
        temp_concat.append(Concatenate(axis=2)(custom_avg_pool_ary[row]))
    custom_avg_pool = Concatenate(axis=1)(temp_concat)

    temp_concat = []
    for row in range(0, GRID_ROW):
        temp_concat.append(Concatenate(axis=2)(custom_max_pool_ary[row]))
    custom_max_pool = Concatenate(axis=1)(temp_concat)

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)

    assert avg_pool._keras_shape[-1] == 1
    assert custom_avg_pool._keras_shape[-1] == 1
    assert max_pool._keras_shape[-1] == 1
    assert custom_max_pool._keras_shape[-1] == 1

    concat = Concatenate(axis=3)([avg_pool, max_pool])
    custom_concat = Concatenate(axis=3)([custom_avg_pool, custom_max_pool])

    assert concat._keras_shape[-1] == 2
    assert custom_concat._keras_shape[-1] == 2

    print("cbam_feature:", cbam_feature)
    print("avg_pool:", avg_pool)
    print("custom_avg_pool:", custom_avg_pool)
    print("max_pool:", max_pool)
    print("custom_max_pool:", custom_max_pool)
    print("concat:", concat)
    print("custom_concat:", custom_concat)

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(custom_concat)

    print("convcbam_feature:", cbam_feature)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    # output = Lambda(lambda x: tf.multiply(x[0], x[1]))([input_feature, cbam_feature])
    # return output
    return multiply([input_feature, cbam_feature])


def channel_attention_custom(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])
