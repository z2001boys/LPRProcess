from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import conv_bn_relu
def GetMdl( ClassNum ):

    input = Input(shape=(100, 100, 2))

    conv1 = conv_bn_relu(filters=32, kernel_size=(3, 3))(input)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = conv_bn_relu(filters=64, kernel_size=(3, 3))(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3a = conv_bn_relu(filters=128, kernel_size=(3, 3))(pool2)
    conv3b = conv_bn_relu(filters=64, kernel_size=(1, 1))(conv3a)
    conv3c = conv_bn_relu(filters=128, kernel_size=(3, 3))(conv3b)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3c)

    conv4a = conv_bn_relu(filters=256, kernel_size=(3, 3))(pool3)
    conv4b = conv_bn_relu(filters=128, kernel_size=(1, 1))(conv4a)
    conv4c = conv_bn_relu(filters=256, kernel_size=(3, 3))(conv4b)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4c)

    conv5a = conv_bn_relu(filters=512, kernel_size=(3, 3))(pool4)
    conv5b = conv_bn_relu(filters=256, kernel_size=(1, 1))(conv5a)
    conv5c = conv_bn_relu(filters=512, kernel_size=(3, 3))(conv5b)
    conv5d = conv_bn_relu(filters=256, kernel_size=(1, 1))(conv5c)
    conv5e = conv_bn_relu(filters=512, kernel_size=(3, 3))(conv5d)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv5e)

    conv6a = conv_bn_relu(filters=1024, kernel_size=(3, 3))(pool5)
    conv6b = conv_bn_relu(filters=512, kernel_size=(1, 1))(conv6a)
    conv6c = conv_bn_relu(filters=1024, kernel_size=(3, 3))(conv6b)
    conv6d = conv_bn_relu(filters=512, kernel_size=(1, 1))(conv6c)
    conv6e = conv_bn_relu(filters=1024, kernel_size=(3, 3))(conv6d)

    conv7 = Conv2D(filters=ClassNum, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6e)
    pool7 = GlobalAveragePooling2D()(conv7)
    output = Activation('softmax')(pool7)

    model = Model(inputs=input, outputs=output)

    return model