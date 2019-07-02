from keras.models import Model
from keras.layers import Activation, Input,ReLU, Lambda, Flatten, Dense, Reshape, Permute, SeparableConv2D, SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import random_normal, constant, glorot_uniform, glorot_normal, random_uniform
from keras.layers.merge import Concatenate

import tensorflow as tf
#from testing.non_local import non_local_block
from keras import backend as K
from keras.layers.merge import Multiply, Add
import numpy as np
from keras.applications import MobileNetV2
import re
from keras.layers import BatchNormalization, DepthwiseConv2D

def relu(x): return Activation('relu')(x)
def elu(x): return Activation('elu')(x)

def linear(x): return Activation('linear')(x)

def tanh(x, name=[]): return Activation('tanh',name=name)(x)

def sigmoid(x): return Activation('sigmoid')(x)

def leaky(x): return LeakyReLU(alpha=0.1)(x)

def conv(x, nf, ks, name, weight_decay, dilation_rate=1):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None #TODO: check if l2 or batch norm
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer = glorot_normal(seed=None), #TODO: Check this!
               #kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0),
               dilation_rate=(dilation_rate,dilation_rate))(x)
    return x

def convKoef(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None #TODO: check if l2 or batch norm
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               #kernel_initializer = glorot_normal(seed=None), #TODO: Check this!
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def convKONF(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None #TODO: check if l2 or batch norm
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               #kernel_initializer = glorot_normal(seed=None), #TODO: Check this!
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(-np.log((1-0.01)/0.01)))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

n_classes = 3

def net_model(x, config): # for black and white picture debuging
    # Block 1
    x = conv(x, 8, 3, "conv1_1", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")  # 32 # 184

    # Block 2
    x = conv(x, 16, 3, "conv2_2", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = pooling(x, 2, 2, "pool2_4")  # 16 # 92

    x = conv(x, 16, 5, "conv2_3", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = pooling(x, 2, 2, "pool2_1")  # 16 # 46

    x = conv(x, 16, 5, "conv2_5", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_2")  # 8 # 23

    x = conv(x, 64, 5, "conv2_7", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 64, 7, "conv2_79", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 128, 7, "conv2_10", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    return x

def net_model_noLane(x, config): # for black and white picture debuging
    # Block 1
    x = conv(x, 8, 3, "conv1_1", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")  # 32 # 184

    # Block 2
    x = conv(x, 16, 3, "conv2_2", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = pooling(x, 2, 2, "pool2_4")  # 16 # 92

    x = conv(x, 16, 5, "conv2_3", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = pooling(x, 2, 2, "pool2_1")  # 16 # 46

    x = conv(x, 16, 5, "conv2_5", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_2")  # 8 # 23

    x = conv(x, 64, 5, "conv2_7", (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 128, 7, "Mconv5_stage%d_L" % (9), (config.weight_decay, 0))
    x_end = leaky(x)

    # x_a = sigmoid(x_a)
    # x_b = convKoef(x, config.num_prediction_cells, 1, "conv_b", (config.weight_decay, 0))
    # x_b = sigmoid(x_b)
    # x_c = convKoef(x, config.num_prediction_cells, 1, "conv_c", (config.weight_decay, 0))
    # x_c = sigmoid(x_c)
    # x_conf = convKoef(x, config.num_prediction_cells, 1, "conv_conf", (config.weight_decay, 0))
    # x_conf =  x_conf
    x_seg = convKoef(x_end, 8, 1, "Segmentation_head", (config.weight_decay, 0))
    x = convKoef(Concatenate()([x_seg, x_end]), 4 * config.num_prediction_cells, 1, "Object_detection_head",
                 (config.weight_decay, 0))
    # x = Concatenate()([x_a, x_conf])
    x_seg = Reshape([config.grid_size, config.grid_size, 8], name="segmentation")(x_seg)
    x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x)

    def slice1(x):
        return x[:, :, :, :, 0:3]

    def slice2(x):
        return tf.keras.backend.expand_dims(x[:, :, :, :, 3], axis=-1)

    coord = Lambda(slice1)(x)
    conf = Lambda(slice2)(x)
    coord = tanh(coord)
    x = Concatenate(name="object_detection")([coord, conf])

    # x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x)
    return [x, x_seg]

def slice1(x):
    return x[:, :, :, :, 0:3]

def slice2(x):
    return tf.keras.backend.expand_dims(x[:, :, :, :, 3], axis=-1)

def stage1_block(x, branch, config):
    # Block 1
  #  x = conv(x, 32, 1, "Mconv1_stage1_L%d" % branch, (config.weight_decay, 0))
  #  x = BatchNormalization()(x)
  #  x = relu(x)
 #   x = conv(x, 128, 5, "Mconv1_stage1_L%d" % branch, (config.weight_decay, 0))
 #   x = BatchNormalization()(x)
 #   x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 512, 1, "Mconv3_stage1_L%d" % branch, (config.weight_decay, 0))
  #  x = BatchNormalization()(x)
    x = relu(x)


 #   x = SCNN_D_prep(x, name=(0, 0))
 #   x = SCNN_D(x, name=(0, 1))

    x_end = convKoef(x, 4 * config.num_prediction_cells, 1, "Mconv5_stage1_L%d" % branch, (config.weight_decay, 0))

    x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x_end)
    coord = Lambda(slice1)(x)
    conf = Lambda(slice2)(x)
    coord = tanh(coord)
    x = Concatenate()([coord, conf])
    return Reshape([config.grid_size, config.grid_size, config.num_prediction_cells * 4],
                   name="output_stage1_L%d" % branch)(x)

    return x

def SCNN_D_prep(x, name=(0,0)):
    name1, name2 = name
    x_shape = K.int_shape(x)
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if len(x_shape) == 4:  # spatial / image data

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = x_shape
        else:
            batchsize, dim1, dim2, channels = x_shape

    sliceD = []
    #for i in range(0, dim1):
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 0, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 1, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 2, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 3, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 4, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 5, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 6, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 7, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 8, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 9, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 10, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 11, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 12, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 13, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 14, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 15, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 16, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 17, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 18, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 19, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 20, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 21, :, :], axis=1), output_shape=(1,23,channels))(x))
    sliceD.append(Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 22, :, :], axis=1), output_shape=(1,23,channels))(x))



    output_i_ = sliceD[0]
    #output_hole_D= output_i_
    list_test = []
    list_test.append(output_i_)
   # output_i_ = sliceD[1]
    for i in range(0, dim1 -1):
       i_Input =  Conv2D(channels, (1, 9), padding='same', use_bias=True,
               name=('SCNN_D_prep_%d_%d_D_%d' % (name1, i, name2)), kernel_initializer=random_normal(stddev=0.0187))(output_i_)
       i_Input = leaky(i_Input)
       output_i_ = Add()([sliceD[i+1],i_Input])
       list_test.append(output_i_)
      # output_hole_D = Concatenate(axis=1)([output_hole_D, output_i_])

    #output_hole_D = Concatenate(axis=1)(list_test)

    output_i_U = list_test[-1]
    list_test_D = []
    list_test_D.append(output_i_U)
    for i in reversed(range(0, dim1-1)):
        i_Input =  Conv2D(channels, (1, 9), padding='same', use_bias=True,
               name=('SCNN_D_prep_%d_%d_U_%d' % (name1, i, name2)), kernel_initializer=random_normal(stddev=0.0187))(output_i_U)
        i_Input = leaky(i_Input)
    #    output_i_U
        output_i_U = Add()([list_test[i],  i_Input])
        list_test_D.append(output_i_U)
    output_hole_U = Concatenate(axis=1)(list_test_D)
    #output_hole_U = non_local_block(x, compression=2, name=SCNN_D_prep_%d_%d_U_%d' % (name1, i, name2))
     #   list_test_D.append(output_i_U)

    #K.eval(output_i_)

    return output_hole_U

def SCNN_D(x,weight_decay, name=(0,0)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None  # TODO: check if l2 or batch norm
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    name1, name2= name
    x_shape = K.int_shape(x)
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if len(x_shape) == 4:  # spatial / image data

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = x_shape
        else:
            batchsize, dim1, dim2, channels = x_shape

    conv2d_shared_D = Conv2D(channels, (1, 3), padding='same', use_bias=True,
               name=('SCNN_D_%d_%d' % (name1, name2)), kernel_initializer=random_normal(stddev=0.0187),kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg)

    conv2d_shared_U = Conv2D(channels, (1, 3), padding='same', use_bias=True,
                             name=('SCNN_D_%d_%d_U' % (name1, name2)), kernel_initializer=random_normal(stddev=0.0187),kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg)

    sliceD = []
    #for i in range(0, dim1):
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 0, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 1, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 2, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 3, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 4, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 5, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 6, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 7, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 8, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 9, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 10, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 11, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 12, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 13, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 14, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 15, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 16, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 17, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 18, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 19, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 20, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 21, :, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, 22, :, :], axis=1), output_shape=(1, 23, channels))(x))



    output_i_ = sliceD[0]
    #output_hole_D= output_i_
    list_test = []
    list_test.append(output_i_)
   # output_i_ = sliceD[1]
    for i in range(1, dim1):
       i_Input = conv2d_shared_D(output_i_)
       i_Input = elu(i_Input)
       output_i_ = Add()([sliceD[i],i_Input])
       list_test.append(output_i_)
      # output_hole_D = Concatenate(axis=1)([output_hole_D, output_i_])

    #output_hole_D = Concatenate(axis=1)(list_test)

    output_i_U = list_test[-1]
    list_test_D = []
    list_test_D.append(output_i_U)
    for i in reversed(range(0, dim1-1)):
        i_Input = conv2d_shared_U(output_i_U)
        i_Input = elu(i_Input)
    #    output_i_U
        output_i_U = Add()([list_test[i],  i_Input])
        list_test_D.append(output_i_U)
    output_hole_U = Concatenate(axis=1)(list_test_D)
     #   list_test_D.append(output_i_U)

    #K.eval(output_i_)

    return output_hole_U

def SCNN_R(x,weight_decay, name=(0,0)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None  # TODO: check if l2 or batch norm
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    name1, name2= name
    x_shape = K.int_shape(x)
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if len(x_shape) == 4:  # spatial / image data

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = x_shape
        else:
            batchsize, dim1, dim2, channels = x_shape

    conv2d_shared_R = Conv2D(channels, (3, 1), padding='same', use_bias=True,
               name=('SCNN_D_%d_%d_R' % (name1, name2)), kernel_initializer=random_normal(stddev=0.0187),kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg)

    conv2d_shared_L = Conv2D(channels, (3, 1), padding='same', use_bias=True,
                             name=('SCNN_D_%d_%d_L' % (name1, name2)), kernel_initializer=random_normal(stddev=0.0187),kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg)

    sliceD = []
    #for i in range(0, dim1):
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :, 0, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:,:, 1, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,2, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,3, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,4, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,5, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,6, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,7, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,8, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,9, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,10, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,11, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,12, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,13, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,14, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,15, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,16, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,17, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,18, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,19, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,20, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,21, :], axis=1), output_shape=(1, 23, channels))(x))
    sliceD.append(
        Lambda(lambda x: tf.keras.backend.expand_dims(x[:, :,22, :], axis=1), output_shape=(1, 23, channels))(x))



    output_i_ = sliceD[0]
    #output_hole_D= output_i_
    list_test = []
    list_test.append(output_i_)
   # output_i_ = sliceD[1]
    for i in range(1, dim1):
       i_Input = conv2d_shared_R(output_i_)
       i_Input = leaky(i_Input)
       output_i_ = Add()([sliceD[i],i_Input])
       list_test.append(output_i_)
      # output_hole_D = Concatenate(axis=1)([output_hole_D, output_i_])

    #output_hole_D = Concatenate(axis=1)(list_test)

    output_i_U = list_test[-1]
    list_test_D = []
    list_test_D.append(output_i_U)
    for i in reversed(range(0, dim1-1)):
        i_Input = conv2d_shared_L(output_i_U)
        i_Input = leaky(i_Input)
    #    output_i_U
        output_i_U = Add()([list_test[i],  i_Input])
        list_test_D.append(output_i_U)
    output_hole_U = Concatenate(axis=1)(list_test_D)
     #   list_test_D.append(output_i_U)

    #K.eval(output_i_)

    return output_hole_U

def stageT_block(x, stage, branch, config):


  #  x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
  #  x = BatchNormalization()(x)
  #  x = relu(x)

  #  x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
  #  x = BatchNormalization()(x)
  #  x = relu(x)

    x = conv(x, 256, 3, "Mconv3_stage%d_L%d" % (stage, branch), (config.weight_decay, 0), dilation_rate=1)
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 256, 3, "Mconv4_stage%d_L%d" % (stage, branch), (config.weight_decay, 0), dilation_rate=1)
    x = BatchNormalization()(x)
    x = relu(x)


    x = conv(x, 256, 3, "Mconv5_stage%d_L%d" % (stage, branch), (config.weight_decay, 0), dilation_rate=1)
    x = BatchNormalization()(x)
    x = relu(x)


    x = conv(x, 256, 3, "Mconv6_stage%d_L%d" % (stage, branch), (config.weight_decay, 0), dilation_rate=1)
    x = BatchNormalization()(x)
    x = relu(x)


    x = conv(x, 512, 1, "Mconv7_stage%d_L%d" % (stage, branch), (config.weight_decay, 0), dilation_rate=1)
    x = relu(x)


    x_end = convKoef(x, 4 * config.num_prediction_cells, 1, "Mconv9_stage1_L%d" % branch, (config.weight_decay, 0))

    x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x_end)
    coord = Lambda(slice1)(x)
    conf = Lambda(slice2)(x)
    coord = tanh(coord)
    x = Concatenate()([coord, conf])
    return Reshape([config.grid_size, config.grid_size, config.num_prediction_cells * 4],
                   name="output_stage%d_L%d" % (stage, branch))(x)
    #return x

def stageT_plus_block(x, stage, branch, config):


  #  x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
  #  x = BatchNormalization()(x)
  #  x = relu(x)

  #  x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
  #  x = BatchNormalization()(x)
  #  x = relu(x)



    x = conv(x, 128, 5, "Mconv3_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    x = conv(x, 128, 5, "Mconv4_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    x = conv(x, 128, 5, "Mconv5_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)



# x = conv(x, 256, 3, "Mconv6_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
   # x = BatchNormalization()(x)
   # x = elu(x)


    x = conv(x, 1024, 1, "Mconv7_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = relu(x)

    x_end = convKoef(x, 4 * config.num_prediction_cells, 1, "Mconv9_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))

    x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x_end)
    coord = Lambda(slice1)(x)
    conf = Lambda(slice2)(x)
    coord = tanh(coord)
    x = Concatenate()([coord, conf])
    return Reshape([config.grid_size,config.grid_size,config.num_prediction_cells*4], name="output_stage%d_L%d" % (stage, branch))(x)


def stageKONF_block(x, stage, branch, config):

    x = conv(x, 128, 3, "Mconv1_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 128, 3, "Mconv3_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)


    x = conv(x, 256, 3, "Mconv4_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 512, 1, "Mconv5_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = relu(x)

    x_end = convKONF(x, 1 * config.num_prediction_cells, 1, "output_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))

    return x_end

def stageLOK_block(x, stage, branch, config):

    x = conv(x, 128, 3, "Mconv1_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)

    x = conv(x, 128, 3, "Mconv3_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)


    x = conv(x, 256, 3, "Mconv4_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)


    x = conv(x, 512, 1, "Mconv5_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = relu(x)


    x_end = convKoef(x, 3 * config.num_prediction_cells, 1, "Mconv6_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    return tanh(x_end, name=  "output_stage%d_L%d" % (stage, branch))



def stageKONF_block_plus(x, stage, branch, config):

    x = conv(x, 128, 3, "Mconv1_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    x = conv(x, 128, 3, "Mconv3_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)


    #x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    #x = BatchNormalization()(x)
    #x = relu(x)

    x = conv(x, 512, 1, "Mconv5_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = relu(x)

    x_end = convKONF(x, 1 * config.num_prediction_cells, 1, "output_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))

    return x_end

def stageLOK_block_plus(x, stage, branch, config):
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = BatchNormalization()(x)
    x = elu(x)

    # x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    # x = BatchNormalization()(x)
    # x = relu(x)

    x = conv(x, 512, 1, "Mconv5_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = relu(x)

    x_end = convKoef(x, 3 * config.num_prediction_cells, 1, "Mconv6_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    return tanh(x_end, name=  "output_stage%d_L%d" % (stage, branch))


def stage_Mobile(x, stage, branch, config):
    # Block 1
    x = bottleneck_block(x, "Mconv0_stage%d_L%d" % (stage, branch), add=False, first_in=256, expand=256)
    x = bottleneck_block(x, "Mconv1_stage%d_L%d" % (stage, branch))
    x = bottleneck_block(x, "Mconv2_stage%d_L%d" % (stage, branch))
    x = bottleneck_block(x, "Mconv3_stage%d_L%d" % (stage, branch))
    x = bottleneck_block(x, "Mconv4_stage%d_L%d" % (stage, branch))
    x = bottleneck_block(x, "Mconv5_stage%d_L%d" % (stage, branch))
    x = bottleneck_block(x, "Mconv6_stage%d_L%d" % (stage, branch))
    x = bottleneck_block(x, "Mconv7_stage%d_L%d" % (stage, branch))



    x = conv(x, 512, 1, "Mconv8_stage%d_L%d" % (stage, branch), (config.weight_decay, 0))
    x = relu(x)

    x_end = convKoef(x, 4 * config.num_prediction_cells, 1, "Mconv9_stage1_L%d" % branch, (config.weight_decay, 0))

    x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x_end)
    coord = Lambda(slice1)(x)
    conf = Lambda(slice2)(x)
    coord = tanh(coord)
    x = Concatenate()([coord, conf])
    return Reshape([config.grid_size, config.grid_size, config.num_prediction_cells * 4],
                   name="output_stage%d_L%d" % (stage, branch))(x)


def get_lrmult(model):

    # setup lr multipliers for conv layers
    lr_mult = dict()

    for layer in model.layers:

        if isinstance(layer, Conv2D):


            if re.match("Mconv\d_stage1.*", layer.name) or re.match("Mconv\d_stage0.*", layer.name) or re.match("output_stage1_L1.*", layer.name):
                kernel_name = layer.weights[0].name
         #       bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1.
         #       lr_mult[bias_name] = 2.

            # stage > 1
            elif re.match("Mconv\d_stage2.*", layer.name) or re.match("output_stage2_L1.*", layer.name) :
                kernel_name = layer.weights[0].name
             #   bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 2
             #   lr_mult[bias_name] =4
            elif re.match("Mconv\d_stage3.*", layer.name):
                kernel_name = layer.weights[0].name
             #   bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 2.05
            #    lr_mult[bias_name] = 2.05
            elif re.match("Mconv\d_stage4.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 2.1
                lr_mult[bias_name] = 2.1
            elif re.match("Mconv\d_stage5.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 10
                lr_mult[bias_name] = 10
            elif re.match("Mconv\d_stage6.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 12
                lr_mult[bias_name] = 12

         #   elif re.match("SCNN_D_0.*", layer.name ) or re.match("SCNN_D_prep_0.*", layer.name):
         #       print("SETTING non local", layer.name)
         #       kernel_name = layer.weights[0].name
         #      # bias_name = layer.weights[1].name
         #       lr_mult[kernel_name] = 1
            elif re.match("SCNN_D_1.*", layer.name) or re.match("SCNN_D_prep_1.*", layer.name):
                print("SETTING non local", layer.name)
                kernel_name = layer.weights[0].name
#                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 1
            elif re.match("SCNN_D_2.*", layer.name) or re.match("SCNN_D_prep_2.*", layer.name):
                print("SETTING non local", layer.name)
                kernel_name = layer.weights[0].name
               # bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 2
             #   lr_mult[bias_name] = 4
            elif re.match("SCNN_D_3.*", layer.name) or re.match("SCNN_D_prep_3.*", layer.name):
                print("SETTING non local", layer.name)
                kernel_name = layer.weights[0].name
         #       bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 2.05
             #   lr_mult[bias_name] = 7
            elif re.match("SCNN_D_4.*", layer.name) or re.match("SCNN_D_prep_4.*", layer.name):
                print("SETTING non local", layer.name)
                kernel_name = layer.weights[0].name
               # bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 8
               # lr_mult[bias_name] = 5
            elif re.match("SCNN_D_5.*", layer.name) or re.match("SCNN_D_prep_5.*", layer.name):
                print("SETTING non local", layer.name)
                kernel_name = layer.weights[0].name
               # bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 10
               # lr_mult[bias_name] = 5
            elif re.match("SCNN_D_6.*", layer.name) or re.match("SCNN_D_prep_6.*", layer.name):
                print("SETTING non local", layer.name)
                kernel_name = layer.weights[0].name
               # bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 12
               # lr_mult[bias_name] = 5
            # vgg
            else:
               print("matched as vgg layer", layer.name)
               kernel_name = layer.weights[0].name
            #   bias_name = layer.weights[1].name
               lr_mult[kernel_name] = 1.
            #   lr_mult[bias_name] = 2.

    return lr_mult

def vgg_model(x, config):
    # Block 1 # original input size 368
    x = conv(x, 64, 3, "conv1_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1") # size 184
    # Block 2
    x = conv(x, 128, 3, "conv2_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1") # size 92

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1") # size 46
    # Block 4
    x = conv(x, 512, 3, "conv4_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (config.weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = pooling(x, 2, 2, "pool3_2")  # size 23
    x = conv(x, 256, 3, "conv_head_1_CPM", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv_head_2_CPM", (config.weight_decay, 0))
    x = relu(x)

    x = conv(x, 128, 3, "Mconvt_stage%d_L" % (1), (config.weight_decay, 0))
    x = BatchNormalization(axis=-1, name='batch_' + '1_')(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconvt_stage%d_L" % (2), (config.weight_decay, 0))
    x = BatchNormalization(axis=-1, name='batch_' + '2_')(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconvt_stage%d_L" % (3), (config.weight_decay, 0))
    x = BatchNormalization(axis=-1, name='batch_' + '3_')(x)
    x = relu(x)
    x = conv(x, 512, 1, "Mconvt_stage%d_L" % (4), (config.weight_decay, 0))
    x = BatchNormalization(axis=-1, name='batch_' + '4_')(x)
    x = relu(x)
    # Additional non vgg layers
  #  x = conv(x, 128, 7, "Mconv1_stage%d_L" % (1), (config.weight_decay, 0))
  #  x = BatchNormalization(axis=-1, name='batch_' + '5_')(x)
  #  x = leaky(x)
  #  x = conv(x, 128, 7, "Mconv2_stage%d_L" % (2), (config.weight_decay, 0))
  #  x = BatchNormalization(axis=-1, name='batch_' + '6_')(x)
  #  x = leaky(x)
  #  x = conv(x, 128, 7, "Mconv3_stage%d_L" % (3), (config.weight_decay, 0))
  #  x = BatchNormalization(axis=-1, name='batch_' + '7_')(x)
  #  x = leaky(x)
  #  x = conv(x, 128, 7, "Mconv4_stage%d_L" % (4), (config.weight_decay, 0))
  #  x = BatchNormalization(axis=-1, name='batch_' + '8_')(x)
  #  x = leaky(x)
  #  x = conv(x, 128, 7, "Mconv5_stage%d_L" % (5), (config.weight_decay, 0))
  #  x = BatchNormalization(axis=-1, name='batch_' + '9_')(x)
  #  x = leaky(x)
  #  x = conv(x, 128, 7, "Mconv5_stage%d_L" % (6), (config.weight_decay, 0))
  #  x = BatchNormalization(axis=-1, name='batch_' + '10_')(x)
  #  x = leaky(x)
  #  x = conv(x, 128, 7, "Mconv4_stage%d_L" % (7), (config.weight_decay, 0)) # bigger with this 3 stages
  #  x = BatchNormalization(axis=-1, name='batch_' + '11_')(x)
  #  x = leaky(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L" % (8), (config.weight_decay, 0))
    x = BatchNormalization(axis=-1, name='batch_' + '12_')(x)
    x = leaky(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L" % (9), (config.weight_decay, 0))
    x_end = leaky(x)


   # x_a = sigmoid(x_a)
    #x_b = convKoef(x, config.num_prediction_cells, 1, "conv_b", (config.weight_decay, 0))
    #x_b = sigmoid(x_b)
    #x_c = convKoef(x, config.num_prediction_cells, 1, "conv_c", (config.weight_decay, 0))
    #x_c = sigmoid(x_c)
    #x_conf = convKoef(x, config.num_prediction_cells, 1, "conv_conf", (config.weight_decay, 0))
    # x_conf =  x_conf
    x_seg = convKoef(x_end, 8, 1, "Segmentation_head", (config.weight_decay, 0))
    x = convKoef(Concatenate()([x_seg, x_end]), 4 * config.num_prediction_cells, 1, "Object_detection_head", (config.weight_decay, 0))
    #x = Concatenate()([x_a, x_conf])
    x_seg = Reshape([config.grid_size, config.grid_size, 8], name = "segmentation" )(x_seg)
    x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x)

    def slice1(x):
        return x[:, :, :, :, 0:3]

    def slice2(x):
        return  tf.keras.backend.expand_dims(x[:, :, :, :, 3], axis=-1)

    coord = Lambda(slice1)(x)
    conf = Lambda(slice2)(x)
    coord = tanh(coord)
    x = Concatenate(name = "object_detection")([coord, conf])

    #x = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(x)
    return [x, x_seg]

def bottleneck_block(x,name,add=True,first_in= 256, expand = 256):

  m = Conv2D(first_in, (1,1), padding='same',
                          use_bias=False,
                          activation=None,
                          name=name+'1')(x)
  m = BatchNormalization( epsilon=1e-3, momentum=0.999)(m)
  m = ReLU(6.,)(m)
  m = DepthwiseConv2D((3,3), activation=None,
                               use_bias=False,
                            name=name+ '2',padding='same')(m)
  m = BatchNormalization( epsilon=1e-3, momentum=0.999)(m)
  m = ReLU(6., )(m)
  m = Conv2D(expand, (1,1), name=name+ '3',padding='same',
                                         use_bias=False,
                                         activation=None,
                                         )(m)
  m = BatchNormalization( epsilon=1e-3, momentum=0.999)(m)
  if add:
    return Add()([m, x])
  return m

def mobile(x, config):
    base_model = MobileNetV2(input_tensor=x, include_top=False,
                             weights='imagenet', alpha=1.4)
    x = base_model.output
    x = bottleneck_block(x, 'conv1_3_CPM_', add=False, first_in=1280, expand=256)
    x = bottleneck_block(x, 'conv2_4_CPM_', add=False, first_in=256, expand=128)
    #  x = conv(x, 256, 3, "conv4_3_CPM", (0, 0))
    #  x = relu(x)
    #  x = BatchNormalization()(x)
    #  x = conv(x, 128, 3, "conv4_4_CPM", (0, 0))
    #  x = relu(x)
    #  x = BatchNormalization()(x)
    return x

def vgg_model_only(x, config):
    # Block 1 # original input size 368
    x = conv(x, 64, 3, "conv1_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1") # size 184
    # Block 2
    x = conv(x, 128, 3, "conv2_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1") # size 92

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1") # size 46
    # Block 4
    x = conv(x, 512, 3, "conv4_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_3", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_4", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool4_1")  # size 46
    x = conv(x, 512, 3, "conv5_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv5_2", (config.weight_decay, 0))
    x = relu(x)

    x = conv(x, 256, 3, "Mconv0_stage0.", (config.weight_decay, 0))
    x = relu(x)
    x = BatchNormalization()(x)
    x = conv(x, 128, 3, "Mconv1_stage0", (config.weight_decay, 0))
    x = relu(x)
    x = BatchNormalization()(x)

    return x

def vgg16_model_only(x, config):
    # Block 1 # original input size 368
    x = conv(x, 64, 3, "conv1_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1") # size 184
    # Block 2
    x = conv(x, 128, 3, "conv2_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1") # size 92

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1") # size 46

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_3", (config.weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool4_1")  # size 23
    #Block 5
    x = conv(x, 512, 3, "conv5_1", (config.weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv5_2", (config.weight_decay, 0))
    x = relu(x)

    x = conv(x, 256, 3, "Mconv0_stage0.", (config.weight_decay, 0))
    x = relu(x)
    x = BatchNormalization()(x)
    x = conv(x, 128, 3, "Mconv1_stage0", (config.weight_decay, 0))
    x = relu(x)
    x = BatchNormalization()(x)

    return x

def get_training_model_staged(config):
    img_input_shape = (config.img_w, config.img_h, 3)

    #object_detection_shape  = (config.grid_size, config.grid_size, config.num_prediction_cells, 4)

    img_input = Input(shape=img_input_shape)
    #object_detection_Input = Input(shape=object_detection_shape)
    new_x = []
    outputs = []
    if config.vgg19:
        net_out = vgg_model_only(img_input, config)
    elif config.mobile:
        net_out = mobile(img_input, config)
    else:
        net_out = vgg16_model_only(img_input, config)

    if not config.mobile:
        stage1_branch1_net_out = stageT_block(net_out,1, 1, config)
        outputs.append(stage1_branch1_net_out)
    else:
        stage1_branch1_net_out = stageT_block(net_out,1, 1, config)
        outputs.append(stage1_branch1_net_out)
    new_x.append(stage1_branch1_net_out)

    new_x.append(net_out)

    x = Concatenate()(new_x)
    #last_layer = stage1_branch1_net_out
    for sn in range(2, config.stages + 1):
        new_x = []
        stageT_branch1_net_out = stageT_block(x,sn, 1, config)
        #last_layer = stageT_branch1_net_out
        outputs.append(stageT_branch1_net_out)
        new_x.append(stageT_branch1_net_out)

        if sn < config.stages:
            x = Concatenate()(new_x + [net_out])




    #end_stage = stage_END_block(net_out, 99, config)
    #outputs.append(end_stage)


    model = Model(inputs=img_input, outputs=outputs)
    model.summary()  # show params
    return model

def get_testing_model_staged(config):
    img_input_shape = (config.img_w, config.img_h, 3)

    # object_detection_shape  = (config.grid_size, config.grid_size, config.num_prediction_cells, 4)

    outputs = []

    img_input = Input(shape=img_input_shape)
    # object_detection_Input = Input(shape=object_detection_shape)

    net_out = vgg_model_only(img_input, config)
    new_x = []

    stage1_branch1_net_out = stage1_block(net_out, 1, config)
    outputs.append(stage1_branch1_net_out)
    new_x.append(stage1_branch1_net_out)

    new_x.append(net_out)

    x = Concatenate()(new_x)
    output = stage1_branch1_net_out
    for sn in range(2, config.stages + 1):
        #new_x = []
        stageT_branch1_net_out = stageT_block(x, sn, 1, config)
        outputs.append(stageT_branch1_net_out)
        #new_x.append(stageT_branch1_net_out)
        #new_x.append(net_out)

      #  if sn < config.stages:
      #  x = Concatenate()(new_x)
      #  if sn == config.stages:
      #      output = stageT_branch1_net_out

   # out = stage_END_block(net_out, 99, config)
    #outputs.append(out)

   # out = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 4])(output)
    model = Model(inputs=img_input, outputs=outputs)
 #   model.summary()  # show params
    return model

def get_training_model(config):
    img_input_shape = (config.img_w, config.img_h, 1)
    img_input = Input(shape=img_input_shape)

    net = vgg_model(img_input, config)

    model = Model(inputs=img_input, outputs=net)

    model.summary()  # show params
    return model

def get_train_model_splitted(config):
    img_input_shape = (config.img_w, config.img_h, 3)

    img_input = Input(shape=img_input_shape)

    outputs = []
    if config.vgg19:
        net_out = vgg_model_only(img_input, config)
    else:
        net_out = vgg16_model_only(img_input, config)
    new_x = []

    stage1_branch1LOK_net_out = stageLOK_block(net_out, 1, 0, config)
    stage1_branch1KONF_net_out = stageKONF_block(net_out, 1, 1, config)
    outputs.append(stage1_branch1LOK_net_out)
    outputs.append(stage1_branch1KONF_net_out)

    new_x.append(stage1_branch1LOK_net_out)
    new_x.append(stage1_branch1KONF_net_out)

    new_x.append(net_out)

    x = Concatenate()(new_x)
    #last_layer = stage1_branch1_net_out
    for sn in range(2, config.stages + 1):
        new_x = []
        stage1_branch1LOK_net_out = stageLOK_block_plus(x,sn, 0, config)
        stage1_branch1KONF_net_out = stageKONF_block_plus(x, sn, 1, config)

        outputs.append(stage1_branch1LOK_net_out)
        outputs.append(stage1_branch1KONF_net_out)

        new_x.append(stage1_branch1LOK_net_out)
        new_x.append(stage1_branch1KONF_net_out)

        new_x.append(net_out)

        if sn < config.stages:
           x = Concatenate()(new_x)



    #end_stage = stage_END_block(net_out, 99, config)
    #outputs.append(end_stage)


    model = Model(inputs=img_input, outputs=outputs)
    model.summary()  # show params
    return model

def get_test_model_splitted(config):
    img_input_shape = (config.img_w, config.img_h, 3)

    img_input = Input(shape=img_input_shape)

    outputs = []
    if config.vgg19:
        net_out = vgg_model_only(img_input, config)
    else:
        net_out = vgg16_model_only(img_input, config)
   # new_x = []

    stage1_branch1LOK_net_out = stageLOK_block(net_out, 1, 0, config)
    stage1_branch1KONF_net_out = stageKONF_block(net_out, 1, 1, config)

    stage1_branch1LOK_net_out = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 3])(
        stage1_branch1LOK_net_out)
    stage1_branch1KONF_net_out = Reshape([config.grid_size, config.grid_size, config.num_prediction_cells, 1])(
        stage1_branch1KONF_net_out)


    concated = Concatenate()([stage1_branch1KONF_net_out, stage1_branch1LOK_net_out])
    outputs.append( Reshape([config.grid_size, config.grid_size, config.num_prediction_cells * 4],
                   name="output_stage%d_L%d" % (1, 1))(concated))


    #  new_x.append(stage1_branch1_net_out)

    #  new_x.append(net_out)

    #  x = Concatenate()(new_x)
    # last_layer = stage1_branch1_net_out
    #  for sn in range(2, config.stages + 1):
    #      new_x = []
    #      stageT_branch1_net_out = stageT_block(x,sn, 1, config)
    #      last_layer = stageT_branch1_net_out
    #     outputs.append(stageT_branch1_net_out)
    #      new_x.append(stageT_branch1_net_out)
    #      new_x.append(net_out)

    #     if sn < config.stages:
    #          x = Concatenate()(new_x)



    # end_stage = stage_END_block(net_out, 99, config)
    # outputs.append(end_stage)

    model = Model(inputs=img_input, outputs=outputs)
    model.summary()  # show params
    return model


def get_train_model_no_grid(config):
    img_input_shape = (config.img_w, config.img_h, 3)
    img_input = Input(shape=img_input_shape)

    net = net_model_noLane(img_input, config)

    model = Model(inputs=img_input, outputs=net)

    model.summary()  # show params
    return model

if __name__ == "__main__":
    get_train_model_splitted(0.0001)
