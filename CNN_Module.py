import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Cropping2D 
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add, concatenate, multiply
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU, ReLU


# ResNet Module
def bn_relu(input):
    bn = BatchNormalization()(input)
    return Activation("relu")(bn)

def conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return bn_relu(conv)
    return f

def bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f

def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(1e-4))(input)
    return add([shortcut, residual])

def basic_block(filters, init_strides=(1, 1), first_layer_first_block=False):
    def f(input):
        if first_layer_first_block:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                 strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return shortcut(input, residual)

    return f

def bottleneck(filters, init_strides=(1, 1), first_layer_first_block=False):
    def f(input):
        if first_layer_first_block:
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides, padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                    strides=init_strides)(input)

        conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = bn_relu_conv(filters=filters*4, kernel_size=(1, 1))(conv_3_3)
        return shortcut(input, residual)

    return f

# GoogleNet Module v1 (a.k.a inception v1)
def inception_block(filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    def f(input):
        inception_1x1 = Conv2D(filters=filters_1x1,
                               kernel_size=(1, 1), strides=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(2e-4))(input)

        inception_3x3_reduce = Conv2D(filters=filters_3x3_reduce,
                                      kernel_size=(1, 1), strides=(1, 1),
                                      padding='same', activation='relu',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(2e-4))(input)

        inception_3x3 = Conv2D(filters=filters_3x3,
                               kernel_size=(3, 3), strides=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(2e-4))(inception_3x3_reduce)

        inception_5x5_reduce = Conv2D(filters=filters_5x5_reduce,
                                      kernel_size=(1, 1), strides=(1, 1),
                                      padding='same', activation='relu',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(2e-4))(input)
 
        inception_5x5 = Conv2D(filters=filters_5x5,
                               kernel_size=(5, 5), strides=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(2e-4))(inception_5x5_reduce)

        inception_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                      padding='same')(input)


        inception_pool_proj = Conv2D(filters=filters_pool_proj,
                                     kernel_size=(1, 1), strides=(1, 1),
                                     padding='same', activation='relu',
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=l2(2e-4))(inception_pool)

        inception_output = concatenate([inception_1x1, inception_3x3,
                                        inception_5x5, inception_pool_proj])
        return inception_output

    return f

def inception_block_v1(filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    def f(input):
        inception_1x1 = conv_bn_relu(filters=filters_1x1, kernel_size=(1, 1))(input)

        inception_3x3_reduce = conv_bn_relu(filters=filters_3x3_reduce,
                                            kernel_size=(1, 1))(input)
        inception_3x3 = conv_bn_relu(filters=filters_3x3,
                                     kernel_size=(3, 3))(inception_3x3_reduce)

        inception_5x5_reduce = conv_bn_relu(filters=filters_5x5_reduce,
                                            kernel_size=(1, 1))(input)
        inception_5x5 = conv_bn_relu(filters=filters_5x5,
                                     kernel_size=(5, 5))(inception_5x5_reduce)

        inception_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                      padding='same')(input)
        inception_pool_proj = conv_bn_relu(filters=filters_pool_proj,
                                           kernel_size=(1, 1))(inception_pool)

        inception_output = concatenate([inception_1x1, inception_3x3,
                                        inception_5x5, inception_pool_proj])
        return inception_output
    return f

def stem_v4():
    def f(input):
        x = conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(input)
        x = conv_bn_relu(filters=32, kernel_size=(3, 3), padding='valid')(x)
        x = conv_bn_relu(filters=64, kernel_size=(3, 3))(x)

        branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        branch_1 = conv_bn_relu(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
        x = concatenate([branch_0, branch_1])

        branch_0 = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)
        branch_0 = conv_bn_relu(filters=96, kernel_size=(3, 3), padding='valid')(branch_0)

        branch_1 = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)
        branch_1 = conv_bn_relu(filters=64, kernel_size=(1, 7))(branch_1)
        branch_1 = conv_bn_relu(filters=64, kernel_size=(7, 1))(branch_1)
        branch_1 = conv_bn_relu(filters=96, kernel_size=(3, 3), padding='valid')(branch_1)
        x = concatenate([branch_0, branch_1])

        branch_0 = conv_bn_relu(filters=192, kernel_size=(3, 3), strides=(2 ,2), padding='valid')(x)
        branch_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
        x = concatenate([branch_0, branch_1])
        return x
    return f

def inception_a_v4():
    def f(input):
        branch_0 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
        branch_0 = conv_bn_relu(filters=96, kernel_size=(1, 1))(branch_0)

        branch_1 = conv_bn_relu(filters=96, kernel_size=(1, 1))(input)

        branch_2 = conv_bn_relu(filters=64, kernel_size=(1, 1))(input)
        branch_2 = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch_2)

        branch_3 = conv_bn_relu(filters=64, kernel_size=(1, 1))(input)
        branch_3 = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch_3)
        branch_3 = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch_3)
        x = concatenate([branch_0, branch_1, branch_2, branch_3])
        return x
    return f

def inception_b_v4():
    def f(input):
        branch_0 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
        branch_0 = conv_bn_relu(filters=128, kernel_size=(1, 1))(branch_0)

        branch_1 = conv_bn_relu(filters=384, kernel_size=(1, 1))(input)

        branch_2 = conv_bn_relu(filters=192, kernel_size=(1, 1))(input)
        branch_2 = conv_bn_relu(filters=224, kernel_size=(7, 1))(branch_2)
        branch_2 = conv_bn_relu(filters=256, kernel_size=(1, 7))(branch_2)

        branch_3 = conv_bn_relu(filters=192, kernel_size=(1, 1))(input)
        branch_3 = conv_bn_relu(filters=192, kernel_size=(7, 1))(branch_3)
        branch_3 = conv_bn_relu(filters=224, kernel_size=(1, 7))(branch_3)
        branch_3 = conv_bn_relu(filters=224, kernel_size=(7, 1))(branch_3)
        branch_3 = conv_bn_relu(filters=256, kernel_size=(1, 7))(branch_3)
        x = concatenate([branch_0, branch_1, branch_2, branch_3])
        return x 
    return f

def inception_c_v4():
    def f(input):
        branch_0 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
        branch_0 = conv_bn_relu(filters=256, kernel_size=(1, 1))(branch_0)

        branch_1 = conv_bn_relu(filters=256, kernel_size=(1, 1))(input)

        branch_2 = conv_bn_relu(filters=384, kernel_size=(1, 1))(input)
        branch_21 = conv_bn_relu(filters=256, kernel_size=(1, 3))(branch_2)
        branch_22 = conv_bn_relu(filters=256, kernel_size=(3, 1))(branch_2)
        branch_2 = concatenate([branch_21, branch_22])

        branch_3 = conv_bn_relu(filters=384, kernel_size=(1, 1))(input)
        branch_3 = conv_bn_relu(filters=448, kernel_size=(3, 1))(branch_3)
        branch_3 = conv_bn_relu(filters=512, kernel_size=(1, 3))(branch_3)
        branch_31 = conv_bn_relu(filters=256, kernel_size=(1, 3))(branch_3)
        branch_32 = conv_bn_relu(filters=256, kernel_size=(3, 1))(branch_3)
        branch_3 = concatenate([branch_31, branch_32])
        x = concatenate([branch_0, branch_1, branch_2, branch_3])
        return x
    return f

def reduction_a_v4():
    def f(input):
        branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)
        branch_1 = conv_bn_relu(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid')(input)

        branch_2 = conv_bn_relu(filters=192, kernel_size=(1, 1))(input)
        branch_2 = conv_bn_relu(filters=224, kernel_size=(3, 3))(branch_2)
        branch_2 = conv_bn_relu(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch_2)
        x = concatenate([branch_0, branch_1, branch_2])
        return x 
    return f

def reduction_b_v4():
    def f(input):
        branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input)
        branch_1 = conv_bn_relu(filters=192, kernel_size=(1, 1))(input)
        branch_1 = conv_bn_relu(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch_1)

        branch_2 = conv_bn_relu(filters=256, kernel_size=(1, 1))(input)
        branch_2 = conv_bn_relu(filters=256, kernel_size=(1, 7))(branch_2)
        branch_2 = conv_bn_relu(filters=320, kernel_size=(7, 1))(branch_2)
        branch_2 = conv_bn_relu(filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch_2)
        x = concatenate([branch_0, branch_1, branch_2])
        return x 
    return f

def stem_inception_resnet_v1():
    def f(input):
        x = Conv2D(32, (3, 3), activation = "relu", strides = (2, 2), padding = "valid")(input)
        x = Conv2D(32, (3, 3), activation = "relu", padding = "valid")(x)
        x = Conv2D(64, (3, 3), activation = "relu", padding = "same")(x)
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = "valid")(x)
        x = Conv2D(80, (1, 1), activation = "relu", padding = "same")(x)
        x = Conv2D(192, (3, 3), activation = "relu", padding = "valid")(x)
        x = Conv2D(256, (3, 3), activation = "relu", strides = (2, 2), padding = "valid")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    return f

def inception_resnet_a_v1(scale_residual = True):
    def f(input):
        ar1 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)

        ar2 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
        ar2 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar2)

        ar3 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
        ar3 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar3)
        ar3 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar3)

        merged = concatenate([ar1, ar2, ar3])
        ar = Conv2D(256, (1, 1), activation = "linear", padding = "same")(merged)
        if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)

        output = add([input, ar])
        output = BatchNormalization()(output)
        output = Activation("relu")(output)

        return output
    return f

def inception_resnet_b_v1(scale_residual = True):
    def f(input):
        br1 = Conv2D(128, (1, 1), activation = "relu", padding = "same")(input)

        br2 = Conv2D(128, (1, 1), activation = "relu", padding = "same")(input)
        br2 = Conv2D(128, (1, 7), activation = "relu", padding = "same")(br2)
        br2 = Conv2D(128, (7, 1), activation = "relu", padding = "same")(br2)

        merged = concatenate([br1, br2])
        br = Conv2D(896, (1, 1), activation = "linear", padding = "same")(merged)
        if scale_residual: br = Lambda(lambda b: b * 0.1)(br)

        output = add([input, br])
        output = BatchNormalization()(output)
        output = Activation("relu")(output)

        return output
    return f

def inception_resnet_c_v1(scale_residual = True):
    def f(input):
        cr1 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)

        cr2 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)
        cr2 = Conv2D(192, (1, 3), activation = "relu", padding = "same")(cr2)
        cr2 = Conv2D(192, (3, 1), activation = "relu", padding = "same")(cr2)

        merged = concatenate([cr1, cr2])
        cr = Conv2D(1792, (1, 1), activation = "linear", padding = "same")(merged)
        if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)

        output = add([input, cr])
        output = BatchNormalization()(output)
        output = Activation("relu")(output)

        return output
    return f

def reduction_resnet_b_v1():
    def f(input):
        rbr1 = MaxPooling2D((3,3), strides = (2,2), padding = "valid")(input)

        rbr2 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
        rbr2 = Conv2D(384, (3, 3), activation = "relu", strides = (2,2))(rbr2)

        rbr3 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
        rbr3 = Conv2D(256, (3, 3), activation = "relu", strides = (2,2))(rbr3)

        rbr4 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
        rbr4 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(rbr4)
        rbr4 = Conv2D(256, (3, 3), activation = "relu", strides = (2,2))(rbr4)

        merged = concatenate([rbr1, rbr2, rbr3, rbr4])
        rbr = BatchNormalization()(merged)
        rbr = Activation("relu")(rbr)

        return rbr
    return f

def stem_inception_resnet_v2():
    def f(input):
        x = Conv2D(32, (3, 3), activation = "relu", strides = (2, 2))(input)
        x = Conv2D(32, (3, 3), activation = "relu")(x)
        x = Conv2D(64, (3, 3), activation = "relu", padding = "same")(x)

        x1 = MaxPooling2D((3, 3), strides = (2, 2))(x)
        x2 = Conv2D(96, (3, 3), activation = "relu", strides = (2, 2))(x)
        x = concatenate([x1, x2])

        x1 = Conv2D(64, (1, 1), activation = "relu", padding = "same")(x)
        x1 = Conv2D(96, (3, 3), activation = "relu")(x1)

        x2 = Conv2D(64, (1, 1), activation = "relu", padding = "same")(x)
        x2 = Conv2D(64, (7, 1), activation = "relu", padding = "same")(x2)
        x2 = Conv2D(64, (1, 7), activation = "relu", padding = "same")(x2)
        x2 = Conv2D(96, (3, 3), activation = "relu", padding = "valid")(x2)
        x = concatenate([x1, x2])

        x1 = Conv2D(192, (3, 3), activation = "relu", strides = (2, 2))(x)
        x2 = MaxPooling2D((3, 3), strides = (2, 2))(x)
        x = concatenate([x1, x2])

        x = bn_relu(x)
        return x
    return f

def inception_resnet_a_v2(scale_residual = True):
    def f(input):
        ar1 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)

        ar2 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
        ar2 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar2)

        ar3 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
        ar3 = Conv2D(48, (3, 3), activation = "relu", padding = "same")(ar3)
        ar3 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(ar3)

        merged = concatenate([ar1, ar2, ar3])
        ar = Conv2D(384, (1, 1), activation = "linear", padding = "same")(merged)

        if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)

        output = add([input, ar])
        output = BatchNormalization()(output)
        output = Activation("relu")(output)
    
        return output
    return f

def inception_resnet_b_v2(scale_residual = True):
    def f(input):
        br1 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)

        br2 = Conv2D(128, (1, 1), activation = "relu", padding = "same")(input)
        br2 = Conv2D(160, (1, 7), activation = "relu", padding = "same")(br2)
        br2 = Conv2D(192, (7, 1), activation = "relu", padding = "same")(br2)

        merged = concatenate([br1, br2])
        br = Conv2D(1152, (1, 1), activation = "linear", padding = "same")(merged)
        if scale_residual: br = Lambda(lambda b: b * 0.1)(br)

        output = add([input, br])
        output = BatchNormalization()(output)
        output = Activation("relu")(output)

        return output
    return f

def inception_resnet_c_v2(scale_residual = True):
    def f(input):
        cr1 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)

        cr2 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)
        cr2 = Conv2D(224, (1, 3), activation = "relu", padding = "same")(cr2)
        cr2 = Conv2D(256, (3, 1), activation = "relu", padding = "same")(cr2)

        merged = concatenate([cr1, cr2])
        cr = Conv2D(2144, (1, 1), activation = "linear", padding = "same")(merged)
        if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)

        output = add([input, cr])
        output = BatchNormalization()(output)
        output = Activation("relu")(output)

        return output
    return f

def reduction_resnet_a(k = 192, l = 224, m = 256, n = 384):
    def f(input):
        rar1 = MaxPooling2D((3,3), strides = (2,2))(input)

        rar2 = Conv2D(n, (3, 3), activation = "relu", strides = (2,2))(input)

        rar3 = Conv2D(k, (1, 1), activation = "relu", padding = "same")(input)
        rar3 = Conv2D(l, (3, 3), activation = "relu", padding = "same")(rar3)
        rar3 = Conv2D(m, (3, 3), activation = "relu", strides = (2,2))(rar3)

        merged = concatenate([rar1, rar2, rar3])
        rar = BatchNormalization()(merged)
        rar = Activation("relu")(rar)

        return rar
    return f

def reduction_resnet_b_v2():
    def f(input):
        rbr1 = MaxPooling2D((3,3), strides = (2,2), padding = "valid")(input)

        rbr2 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
        rbr2 = Conv2D(384, (3, 3), activation = "relu", strides = (2,2))(rbr2)

        rbr3 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
        rbr3 = Conv2D(288, (3, 3), activation = "relu", strides = (2,2))(rbr3)

        rbr4 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
        rbr4 = Conv2D(288, (3, 3), activation = "relu", padding = "same")(rbr4)
        rbr4 = Conv2D(320, (3, 3), activation = "relu", strides = (2,2))(rbr4)

        merged = concatenate([rbr1, rbr2, rbr3, rbr4])
        rbr = BatchNormalization()(merged)
        rbr = Activation("relu")(rbr)

        return rbr
    return f

# DarkNet53 Module
def darknet_bottleneck(filters, init_strides=(1, 1)):
    def f(input):
        conv1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
                             strides=init_strides)(input)
        residual = bn_relu_conv(filters=filters*2, kernel_size=(3, 3),
                                strides=init_strides)(conv1)
        return shortcut(input, residual)
    return f

# DenseNet Module
def conv_block(nb_filter):
    def f(input):
        # 1x1 convolution (Bottleneck layer)
        '''
        x = BatchNormalization(epsilon=1.1e-5,
                               gamma_regularizer=l2(1e-4),
                               beta_regularizer=l2(1e-4))(input)
        '''
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Conv2D(filters=nb_filter*4, kernel_size=(1, 1), strides=(1, 1),
                   padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)

        # 3x3 convolution 
        '''
        x = BatchNormalization(epsilon=1.1e-5,
                               gamma_regularizer=l2(1e-4),
                               beta_regularizer=l2(1e-4))(x)
        '''
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=nb_filter, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        return x
    return f

def dense_block(nb_layer, nb_filter, growth_rate):
    def f(input):
        concat_feature = input
        new_nb_filter = nb_filter

        for i in range(nb_layer):
            x = conv_block(growth_rate)(concat_feature)
            concat_feature = concatenate([concat_feature, x])

            new_nb_filter += growth_rate
        return concat_feature, new_nb_filter
    return f

def transition_block(nb_filter, compression):
    def f(input):
        '''
        x = BatchNormalization(epsilon=1.1e-5,
                               gamma_regularizer=l2(1e-4),
                               beta_regularizer=l2(1e-4))(input)
        '''
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Conv2D(filters=int(nb_filter*compression), kernel_size=(1, 1),
                   strides=(1, 1), padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x
    return f

# SENet Module
def squeeze_excitation_block(out_dim, ratio):
    def f(input):
        squeeze = GlobalAveragePooling2D()(input)

        excitation = Dense(units=out_dim // ratio)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1,1,out_dim))(excitation)

        scale = multiply([input, excitation])
        return scale
    return f

# MobileNet Module
# alpha: width multiplier: 1, 0.75, 0.5, 0.25 
# rho: depth multiplier: 1, 6/7, 5/7, 4/7
def mn_conv_block(filters, alpha, kernel=(3, 3), strides=(1, 1)):
    def f(input):
        new_filters = int(filters*alpha)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(input)
        x = Conv2D(new_filters, kernel, padding='valid', strides=strides)(x)
        x = BatchNormalization()(x)
        return ReLU(6.)(x)
        #return relu(x, max_value=6.)
        #return LeakyReLU(0.3)(x)
    return f

def depthwise_conv_block(pointwise_conv_filters, alpha, rho=1, strides=(1, 1)):
    def f(input):
        new_pointwise_conv_filters = int(pointwise_conv_filters*alpha)

        if strides == (1, 1):
            x = input
        else:
            x = ZeroPadding2D(((0, 1), (0, 1)))(input)
        x = DepthwiseConv2D((3, 3),
                            padding='same' if strides == (1, 1) else 'valid',
                            depth_multiplier=rho,
                            strides=strides,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        #x = relu(x, max_value=6.)
        #x = LeakyReLU(0.3)(x)
        x = ReLU(6.)(x)
        x = Conv2D(new_pointwise_conv_filters, (1, 1),
                   padding='same', strides=(1, 1))(x)
        x = BatchNormalization()(x)
        return ReLU(6.)(x)
        #return relu(x, max_value=6.)
        #return LeakyReLU(0.3)(x)
    return f

# MobileNetv2 Module
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def correct_pad(input, kernel_size):
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(input)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        new_kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (new_kernel_size[0] // 2, new_kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

# expansion(t): 6~10
# width multiplier: 0.35~1.4
# depth multiplier: 96~224
def inverted_res_block(t, strides, alpha, filters, is_expansion=True):
    def f(input):
        in_channels = K.int_shape(input)[-1]
        pointwise_conv_filters = int(filters*alpha)
        pointwise_filters = make_divisible(pointwise_conv_filters, 8)
        x = input

        if is_expansion:
            x = Conv2D(t*in_channels, kernel_size=(1, 1),
                       padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU(6.)(x)

        if strides == (2, 2):
            x = ZeroPadding2D(padding=correct_pad(x, 3))(x)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides,             
                            padding='same' if strides == (1, 1) else 'valid',
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)

        x = Conv2D(pointwise_filters, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

        if in_channels == pointwise_filters and strides == (1, 1):
            return add([input, x])
        return x
    return f

def grouped_conv_block(grouped_channels, cardinality, strides):
    def f(input):
        init = input
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        group_list = []

        if cardinality == 1:
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                       kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(init)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation('relu')(x)
            return x

        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
            if K.image_data_format() == 'channels_last' else
            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                       kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)

            group_list.append(x)

        group_merge = concatenate(group_list, axis=channel_axis)
        x = BatchNormalization(axis=channel_axis)(group_merge)
        x = Activation('relu')(x)

        return x
    return f

def xt_bottleneck_block(filters=64, cardinality=8, strides=(1,1)):
    def f(input):
        init = input

        grouped_channels = int(filters / cardinality)
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if K.image_data_format() == 'channels_first':
            if init.shape[1] != 2 * filters:
                init = Conv2D(filters * 2, (1, 1), padding='same', strides=strides,
                              use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(init)
                init = BatchNormalization(axis=channel_axis)(init)
        else:
            if init.shape[-1] != 2 * filters:
                init = Conv2D(filters * 2, (1, 1), padding='same', strides=strides,
                              use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(init)
                init = BatchNormalization(axis=channel_axis)(init)

        x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(input)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = grouped_conv_block(grouped_channels, cardinality, strides)(x)

        x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization(axis=channel_axis)(x)

        x = add([init, x])
        x = Activation('relu')(x)

        return x 
    return f

def grouped_conv_block_v2(grouped_channels, cardinality, strides):
    def f(input):
        init = input
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        group_list = []

        if cardinality == 1:
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                       kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(init)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation('relu')(x)
            return x

        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
            if K.image_data_format() == 'channels_last' else
            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=strides,
                       kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)

            group_list.append(x)

        group_merge = concatenate(group_list, axis=channel_axis)
        x = BatchNormalization(axis=channel_axis)(group_merge)
        x = Activation('relu')(x)

        return x
    return f
	
def xt_bottleneck_block_v2(filters=64, cardinality=8, strides=(1,1)):
    def f(input):
        init = input

        grouped_channels = int(filters / cardinality)
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if K.image_data_format() == 'channels_first':
            if init.shape[1] != 2 * filters:
                init = Conv2D(filters * 4, (1, 1), padding='same', strides=strides,
                              use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(init)
                init = BatchNormalization(axis=channel_axis)(init)
        else:
            if init.shape[-1] != 2 * filters:
                init = Conv2D(filters * 4, (1, 1), padding='same', strides=strides,
                              use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(init)
                init = BatchNormalization(axis=channel_axis)(init)

        x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(input)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = grouped_conv_block(grouped_channels, cardinality, strides)(x)

        x = Conv2D(filters * 4, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization(axis=channel_axis)(x)

        x = add([init, x])
        x = Activation('relu')(x)

        return x 
    return f

def separable_conv_block(filters, kernel_size=(3, 3), strides=(1, 1), block_id=None):
    def f(ip):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

        with K.name_scope('separable_conv_block_%s' % block_id):
            x = Activation('relu')(ip)
            x = SeparableConv2D(filters, kernel_size, strides=strides,
                                name='separable_conv_1_%s' % block_id, padding='same',
                                use_bias=False, kernel_initializer='he_normal',
                                kernel_regularizer=l2(5e-5))(x)
            x = BatchNormalization(axis=channel_dim, momentum=0.9997,
                                   epsilon=1e-3,
                                   name="separable_conv_1_bn_%s" % (block_id))(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % block_id,                               
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(5e-5))(x)
            x = BatchNormalization(axis=channel_dim, momentum=0.9997,
                                   epsilon=1e-3,
                                   name="separable_conv_2_bn_%s" % (block_id))(x)
        return x
    return f

def adjust_block(filters, block_id=None):
    def f(p, ip):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        img_dim = 2 if K.image_data_format() == 'channels_first' else -2

        with K.name_scope('adjust_block'):
            if p is None:
                p = ip

            elif p.shape[img_dim] != ip.shape[img_dim]:
                with K.name_scope('adjust_reduction_block_%s' % block_id):
                    p = Activation('relu', name='adjust_relu_1_%s' % block_id)(p)

                    p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid',
                                          name='adjust_avg_pool_1_%s' % block_id)(p)
                    p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, 
                                kernel_regularizer=l2(5e-5),
                                name='adjust_conv_1_%s' % block_id,
                                kernel_initializer='he_normal')(p1)

                    p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                    p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                    p2 = AveragePooling2D((1, 1), strides=(2, 2),
                                          padding='valid',
                                          name='adjust_avg_pool_2_%s' % block_id)(p2)
                    p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, 
                                kernel_regularizer=l2(5e-5),
                                name='adjust_conv_2_%s' % block_id,
                                kernel_initializer='he_normal')(p2)

                    p = concatenate([p1, p2], axis=channel_dim)
                    p = BatchNormalization(axis=channel_dim, momentum=0.9997,
                                           epsilon=1e-3,
                                           name='adjust_bn_%s' % block_id)(p)
            elif p.shape[channel_dim] != filters:
                with K.name_scope('adjust_projection_block_%s' % block_id):
                     p = Activation('relu')(p)
                     p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                                name='adjust_conv_projection_%s' % block_id,
                                use_bias=False, kernel_regularizer=l2(5e-5), kernel_initializer='he_normal')(p)
                     p = BatchNormalization(axis=channel_dim, momentum=0.9997, 
                                            epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)
        return p
    return f
	
def normal_cell(filters, block_id=None):
    def f(ip, p):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        
        with K.name_scope('normal_A_block_%s' % block_id):
            p = adjust_block(filters, block_id)(p, ip)
            
            h = Activation('relu')(ip)
            h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                       name='normal_conv_1_%s' % block_id, use_bias=False,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(5e-5))(h)
            h = BatchNormalization(axis=channel_dim, momentum=0.9997, 
                                   epsilon=1e-3, name='normal_bn_1_%s' % block_id)(h)

            with K.name_scope('block_1'):
                x1_1 = separable_conv_block(filters, kernel_size=(5, 5),
                                            block_id='normal_left1_%s' % block_id)(h)
                x1_2 = separable_conv_block(filters,
                                            block_id='normal_right1_%s' % block_id)(p)
                x1 = add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

            with K.name_scope('block_2'):
                x2_1 = separable_conv_block(filters, (5, 5),
                                            block_id='normal_left2_%s' % block_id)(p)
                x2_2 = separable_conv_block(filters, (3, 3),
                                             block_id='normal_right2_%s' % block_id)(p)
                x2 = add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

            with K.name_scope('block_3'):
                x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                      name='normal_left3_%s' % (block_id))(h)
                x3 = add([x3, p], name='normal_add_3_%s' % block_id)

            with K.name_scope('block_4'):
                x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % (block_id))(p)
                x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_right4_%s' % (block_id))(p)
                x4 = add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

            with K.name_scope('block_5'):
                x5 = separable_conv_block(filters, block_id='normal_left5_%s' % block_id)(h)
                x5 = add([x5, h], name='normal_add_5_%s' % block_id)

            x = concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name='normal_concat_%s' % block_id)
        return x, ip
    return f

def reduction_cell(filters, block_id=None):
    def f(ip, p):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

        with K.name_scope('reduction_A_block_%s' % block_id):
            p = adjust_block(filters, block_id)(p, ip)

            h = Activation('relu')(ip)
            h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % block_id,
                       use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-5))(h)
            h = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3,
                                   name='reduction_bn_1_%s' % block_id)(h)

            with K.name_scope('block_1'):
                x1_1 = separable_conv_block(filters, (5, 5), strides=(2, 2),
                                            block_id='reduction_left1_%s' % block_id)(h)
                x1_2 = separable_conv_block(filters, (7, 7), strides=(2, 2),
                                            block_id='reduction_1_%s' % block_id)(p)
                x1 = add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

            with K.name_scope('block_2'):
                x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left2_%s' % block_id)(h)
                x2_2 = separable_conv_block(filters, (7, 7), strides=(2, 2),
                                            block_id='reduction_right2_%s' % block_id)(p)
                x2 = add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

            with K.name_scope('block_3'):
                x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left3_%s' % block_id)(h)
                x3_2 = separable_conv_block(filters, (5, 5), strides=(2, 2),
                                            block_id='reduction_right3_%s' % block_id)(p)
                x3 = add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

            with K.name_scope('block_4'):
                x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % block_id)(x1)
                x4 = add([x2, x4])

            with K.name_scope('block_5'):
                x5_1 = separable_conv_block(filters, (3, 3), block_id='reduction_left4_%s' % block_id)(x1)
                x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_right5_%s' % block_id)(h)
                x5 = add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

            x = concatenate([x2, x3, x4, x5], axis=channel_dim, name='reduction_concat_%s' % block_id)
            return x, ip
    return f

def NasNet(input_shape=None, penultimate_filters=4032, nb_blocks=6, stem_filters=96, initial_reduction=True, skip_reduction_layer_input=True, filters_multiplier=2, dropout=0.5, weight_decay=5e-5):
    input = Input(shape=input_shape)

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    filters = penultimate_filters // 24

    if initial_reduction:
        x = Conv2D(stem_filters, (3, 3), strides=(2, 2), padding='valid',
                   use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(input)
    else:
        x = Conv2D(stem_filters, (3, 3), strides=(1, 1), padding='same',
                   use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(input)

    x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, 
                           name='stem_bn1')(x)

    p = None
    if initial_reduction:
        x, p = reduction_cell(filters // (filters_multiplier ** 2),
                              block_id='stem_1')(x, p)
        x, p = reduction_cell(filters // filters_multiplier, 
                              block_id='stem_2')(x, p)

    for i in range(nb_blocks):
        x, p = normal_cell(filters, block_id='%d' % (i))(x, p)

    x, p0 = reduction_cell(filters * filters_multiplier,
                           block_id='reduce_%d' % (nb_blocks))(x, p)

    p = p0 if not skip_reduction_layer_input else p

    for i in range(nb_blocks):
        x, p = normal_cell(filters * filters_multiplier,
                           block_id='%d' % (nb_blocks + i + 1))(x, p)

    x, p0 = reduction_cell(filters * filters_multiplier ** 2, block_id='reduce_%d' % (2 * nb_blocks))(x, p)

    p = p0 if not skip_reduction_layer_input else p

    for i in range(nb_blocks):
        x, p = normal_cell(filters * filters_multiplier ** 2, block_id='%d' % (2 * nb_blocks + i + 1))(x, p)

    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    output = Dense(34, activation='softmax', kernel_regularizer=l2(weight_decay),
                   name='predictions')(x)

    model = Model(inputs=input, outputs=output)
    return model

# ShuffleNet v1
def channel_shuffle(x, groups):
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, in_channels])
    return x

def group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    if groups == 1:
        return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                      use_bias=False, strides=stride, name=name)(x)

    ig = in_channels // groups
    group_list = []

    assert out_channels % groups == 0

    for i in range(groups):
        offset = i * ig
        group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(x)
        group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride, use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
    return concatenate(group_list, axis=-1, name='%s/concat' % name)

def shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    prefix = 'stage%d/block%d' % (stage, block)

    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    x = group_conv(inputs, in_channels, out_channels=bottleneck_channels, groups=(1 if stage == 2 and block == 1 else groups), name='%s/1x1_gconv_1' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    x = Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False, strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

    x = group_conv(x, bottleneck_channels, out_channels=out_channels if strides == 1 else out_channels - in_channels, groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        ret = add([x, inputs], name='%s/add' % prefix)
    else:
        avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        ret = concatenate([x, avg], bn_axis, name='%s/concat' % prefix)

    ret = Activation('relu', name='%s/relu_out' % prefix)(ret)
    return ret

def shuffle_block(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    x = shuffle_unit(x, in_channels=channel_map[stage - 2], out_channels=channel_map[stage - 1], strides=2,
                     groups=groups, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = shuffle_unit(x, in_channels=channel_map[stage - 1], out_channels=channel_map[stage - 1], strides=1,
                         groups=groups, bottleneck_ratio=bottleneck_ratio, stage=stage, block=(i + 1))
    return x

def ShuffleNet(scale_factor=1.0, pooling='avg', input_shape=(100, 100, 1), groups=3, num_shuffle_units=[3, 7, 3], bottleneck_ratio=0.25, classes=34):
    name = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (scale_factor, groups, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    img_input = Input(shape=input_shape)

    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same',
                   use_bias=False, strides=(2, 2), activation="relu", name="conv1")(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="maxpool1")(x)

    for stage in range(0, len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = shuffle_block(x, out_channels_in_stage, repeat=repeat,
                          bottleneck_ratio=bottleneck_ratio,
                          groups=groups, stage=stage + 2)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name="global_pool")(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name="global_pool")(x)

    x = Dense(classes, name="fc")(x)
    x = Activation('softmax', name='softmax')(x)

    model = Model(inputs=img_input, outputs=x, name=name)

    return model

# ShuffleNet v2
def channel_split_v2(x, name=''):
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle_v2(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

def shuffle_unit_v2(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split_v2(inputs, '%s/spl' % prefix)
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='%s/1x1conv_1' % prefix)(inputs)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_1x1conv_1' % prefix)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same', use_bias=False,name='%s/3x3dwconv' % prefix)(x)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv' % prefix)(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='%s/1x1conv_2' % prefix)(x)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_2' % prefix)(x)
    x = Activation('relu', name='%s/relu_1x1conv_2' % prefix)(x)

    if strides < 2:
        ret = concatenate([x, c_hat], axis=bn_axis, name='%s/concat_1' % prefix)
    else:
        s2 = DepthwiseConv2D(kernel_size=(3,3), strides=2, padding='same', use_bias=False,name='%s/3x3dwconv_2' % prefix)(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_2' % prefix)(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='%s/1x1_conv_3' % prefix)(s2)
        s2 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_3' % prefix)(s2)
        s2 = Activation('relu', name='%s/relu_1x1conv_3'% prefix)(s2)
        ret = concatenate([x, s2], axis=bn_axis, name='%s/concat_2' % prefix)

    ret = Lambda(channel_shuffle_v2, name='%s/channel_shuffle' % prefix)(ret)
    return ret

def shuffle_block_v2(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit_v2(x, out_channels=channel_map[stage-1], strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit_v2(x, out_channels=channel_map[stage-1],strides=1, bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))    
    return x

def ShuffleNetv2(scale_factor=1.0, pooling='max', input_shape=(100,100,1), num_shuffle_units=[3,7,3], bottleneck_ratio=1, classes=34):
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]
    out_channels_in_stage[0] = 24
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    img_input = Input(shape=input_shape)
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2), activation='relu', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = shuffle_block_v2(x, out_channels_in_stage, repeat=repeat, bottleneck_ratio=bottleneck_ratio, stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048

    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    x = Dense(classes, name='fc')(x)
    x = Activation('softmax', name='softmax')(x)

    model = Model(inputs=img_input, outputs=x, name=name)
    return model
