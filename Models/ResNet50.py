from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras import backend as K 
from ILBPLayer import MyLayer
from CNN_Module import bn_relu, conv_bn_relu, bottleneck
def GetMdl( ClassNum ):

    input = Input(shape=(100, 100, 2))

    conv1 = conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    conv2_1 = bottleneck(filters=64, first_layer_first_block=True)(pool1)
    conv2_2 = bottleneck(filters=64)(conv2_1)
    conv2_3 = bottleneck(filters=64)(conv2_2)

    conv3_1 = bottleneck(filters=128, init_strides=(2, 2))(conv2_3)
    conv3_2 = bottleneck(filters=128)(conv3_1)
    conv3_3 = bottleneck(filters=128)(conv3_2)
    conv3_4 = bottleneck(filters=128)(conv3_3)

    conv4_1 = bottleneck(filters=256, init_strides=(2, 2))(conv3_4)
    conv4_2 = bottleneck(filters=256)(conv4_1)
    conv4_3 = bottleneck(filters=256)(conv4_2)
    conv4_4 = bottleneck(filters=256)(conv4_3)
    conv4_5 = bottleneck(filters=256)(conv4_4)
    conv4_6 = bottleneck(filters=256)(conv4_5)

    conv5_1 = bottleneck(filters=512, init_strides=(2, 2))(conv4_6)
    conv5_2 = bottleneck(filters=512)(conv5_1)
    conv5_3 = bottleneck(filters=512)(conv5_2)

    block = bn_relu(conv5_3)

    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                             strides=(1, 1))(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(ClassNum, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)

    model = Model(inputs=input, outputs=dense)

    return model