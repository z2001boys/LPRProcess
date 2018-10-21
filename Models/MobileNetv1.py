from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import mn_conv_block, depthwise_conv_block
def GetMdl( inputShape,ClassNum ):

    input = Input(shape=inputShape)

    x = mn_conv_block(32, alpha=0.75, strides=(2, 2))(input)

    x = depthwise_conv_block(64, alpha=0.75, rho=1)(x)

    x = depthwise_conv_block(128, alpha=0.75, rho=1, strides=(2, 2))(x)

    x = depthwise_conv_block(128, alpha=0.75, rho=1)(x)

    x = depthwise_conv_block(256, alpha=0.75, rho=1, strides=(2, 2))(x)

    x = depthwise_conv_block(256, alpha=0.75, rho=1)(x)

    x = depthwise_conv_block(512, alpha=0.75, rho=1, strides=(2, 2))(x)

    x = depthwise_conv_block(512, alpha=0.75, rho=1)(x)
    x = depthwise_conv_block(512, alpha=0.75, rho=1)(x)
    x = depthwise_conv_block(512, alpha=0.75, rho=1)(x)
    x = depthwise_conv_block(512, alpha=0.75, rho=1)(x)
    x = depthwise_conv_block(512, alpha=0.75, rho=1)(x)

    x = depthwise_conv_block(1024, alpha=0.75, rho=1, strides=(2, 2))(x)

    x = depthwise_conv_block(1024, alpha=0.75, rho=1)(x)

    x = Conv2D(ClassNum, (1, 1), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model