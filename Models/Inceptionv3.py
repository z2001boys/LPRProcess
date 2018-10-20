from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import bn_relu, conv_bn_relu

def GetMdl( inputShape,ClassNum ):

    input = Input(shape=inputShape)

    x = conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(input)
    x = conv_bn_relu(filters=32, kernel_size=(3, 3), padding='valid')(x)
    x = conv_bn_relu(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_bn_relu(filters=80, kernel_size=(1, 1), padding='valid')(x)
    x = conv_bn_relu(filters=192, kernel_size=(3, 3), padding='valid')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 35x35x288: Inception1, 2, 3
    branch1x1 = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)

    branch5x5 = conv_bn_relu(filters=48, kernel_size=(1, 1))(x)
    branch5x5 = conv_bn_relu(filters=64, kernel_size=(5, 5))(branch5x5)

    branch3x3dbl = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn_relu(filters=32, kernel_size=(1, 1))(branch_pool)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # 35x35x288: Inception2
    branch1x1 = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)

    branch5x5 = conv_bn_relu(filters=48, kernel_size=(1, 1))(x)
    branch5x5 = conv_bn_relu(filters=64, kernel_size=(5, 5))(branch5x5)

    branch3x3dbl = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn_relu(filters=64, kernel_size=(1, 1))(branch_pool)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # 35x35x288: Inception3
    branch1x1 = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)

    branch5x5 = conv_bn_relu(filters=48, kernel_size=(1, 1))(x)
    branch5x5 = conv_bn_relu(filters=64, kernel_size=(5, 5))(branch5x5)

    branch3x3dbl = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn_relu(filters=64, kernel_size=(1, 1))(branch_pool)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # 17x17x768: Inception1
    branch3x3 = conv_bn_relu(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)

    branch3x3dbl = conv_bn_relu(filters=64, kernel_size=(1, 1))(x)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3))(branch3x3dbl)
    branch3x3dbl = conv_bn_relu(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch3x3dbl)

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool])

    # 17x17x768: Inception2
    branch1x1 = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)

    branch7x7 = conv_bn_relu(filters=128, kernel_size=(1, 1))(x)
    branch7x7 = conv_bn_relu(filters=128, kernel_size=(1, 7))(branch7x7)
    branch7x7 = conv_bn_relu(filters=192, kernel_size=(7, 1))(branch7x7)

    branch7x7dbl = conv_bn_relu(filters=128, kernel_size=(1, 1))(x)
    branch7x7dbl = conv_bn_relu(filters=128, kernel_size=(7, 1))(branch7x7dbl)
    branch7x7dbl = conv_bn_relu(filters=128, kernel_size=(1, 7))(branch7x7dbl)
    branch7x7dbl = conv_bn_relu(filters=128, kernel_size=(7, 1))(branch7x7dbl)
    branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(1, 7))(branch7x7dbl)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn_relu(filters=192, kernel_size=(1, 1))(branch_pool)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # 17x17x768: Inception3, 4
    for i in range(2):
        branch1x1 = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)

        branch7x7 = conv_bn_relu(filters=160, kernel_size=(1, 1))(x)
        branch7x7 = conv_bn_relu(filters=160, kernel_size=(1, 7))(branch7x7)
        branch7x7 = conv_bn_relu(filters=102, kernel_size=(7, 1))(branch7x7)

        branch7x7dbl = conv_bn_relu(filters=160, kernel_size=(1, 1))(x)
        branch7x7dbl = conv_bn_relu(filters=160, kernel_size=(7, 1))(branch7x7dbl)
        branch7x7dbl = conv_bn_relu(filters=160, kernel_size=(1, 7))(branch7x7dbl)
        branch7x7dbl = conv_bn_relu(filters=160, kernel_size=(7, 1))(branch7x7dbl)
        branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(1, 7))(branch7x7dbl)

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_bn_relu(filters=192, kernel_size=(1, 1))(branch_pool)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # 17x17x768: Inception5
    branch1x1 = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)

    branch7x7 = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)
    branch7x7 = conv_bn_relu(filters=192, kernel_size=(1, 7))(branch7x7)
    branch7x7 = conv_bn_relu(filters=192, kernel_size=(7, 1))(branch7x7)

    branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)
    branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(7, 1))(branch7x7dbl)
    branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(1, 7))(branch7x7dbl)
    branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(7, 1))(branch7x7dbl)
    branch7x7dbl = conv_bn_relu(filters=192, kernel_size=(1, 7))(branch7x7dbl)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn_relu(filters=192, kernel_size=(1, 1))(branch_pool)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # 8x8x1280: Inception1
    branch3x3 = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)
    branch3x3 = conv_bn_relu(filters=320, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch3x3)

    branch7x7x3 = conv_bn_relu(filters=192, kernel_size=(1, 1))(x)
    branch7x7x3 = conv_bn_relu(filters=192, kernel_size=(1, 7))(branch7x7x3)
    branch7x7x3 = conv_bn_relu(filters=192, kernel_size=(7, 1))(branch7x7x3)
    branch7x7x3 = conv_bn_relu(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch7x7x3)

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool])

    # 8x8x1280: Inception2
    for i in range(2):
        branch1x1 = conv_bn_relu(filters=320, kernel_size=(1, 1))(x)
        
        branch3x3 = conv_bn_relu(filters=384, kernel_size=(1, 1))(x)
        branch3x3_1 = conv_bn_relu(filters=384, kernel_size=(1, 3))(branch3x3)
        branch3x3_2 = conv_bn_relu(filters=384, kernel_size=(3, 1))(branch3x3)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2])

        branch3x3dbl = conv_bn_relu(filters=448, kernel_size=(1, 1))(x)
        branch3x3dbl = conv_bn_relu(filters=384, kernel_size=(3, 3))(branch3x3dbl)
        branch3x3dbl_1 = conv_bn_relu(filters=384, kernel_size=(1, 3))(branch3x3dbl)
        branch3x3dbl_2 = conv_bn_relu(filters=384, kernel_size=(3, 1))(branch3x3dbl)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_bn_relu(filters=192, kernel_size=(1, 1))(branch_pool)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool])

    x = GlobalAveragePooling2D()(x)
    output = Dense(ClassNum, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model