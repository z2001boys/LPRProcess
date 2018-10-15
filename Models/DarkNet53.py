from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import bn_relu, conv_bn_relu, darknet_bottleneck
def GetMdl( ClassNum ):

    input = Input(shape=(100, 100, 2))

    x = conv_bn_relu(filters=32, kernel_size=(3, 3))(input)
    x = conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(2, 2))(x)

    x = darknet_bottleneck(filters=32)(x)
    x = conv_bn_relu(filters=128, kernel_size=(3, 3), strides=(2, 2))(x)

    x = darknet_bottleneck(filters=64)(x)
    x = darknet_bottleneck(filters=64)(x)
    x = conv_bn_relu(filters=256, kernel_size=(3, 3), strides=(2, 2))(x)

    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = darknet_bottleneck(filters=128)(x)
    x = conv_bn_relu(filters=512, kernel_size=(3, 3), strides=(2, 2))(x)

    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = darknet_bottleneck(filters=256)(x)
    x = conv_bn_relu(filters=1024, kernel_size=(3, 3), strides=(2, 2))(x)

    x = darknet_bottleneck(filters=512)(x)
    x = darknet_bottleneck(filters=512)(x)
    x = darknet_bottleneck(filters=512)(x)
    x = darknet_bottleneck(filters=512)(x)
    x = darknet_bottleneck(filters=512)(x)
    x = bn_relu(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(ClassNum)(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model