from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from CNN_Module import make_divisible, inverted_res_block,correct_pad
def GetMdl( inputShape ,ClassNum ):

    input = Input(shape=inputShape)

    first_block_filters = make_divisible(32*1, 8)
    x = ZeroPadding2D(padding=correct_pad(input, 3))(input)
    x = Conv2D(first_block_filters, kernel_size=(3, 3),
               strides=(2, 2), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)

    x = inverted_res_block(t=1, strides=(1, 1), alpha=1, filters=16, is_expansion=False)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=24)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=24)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=32)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=32)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=32)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=64)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=64)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=64)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=64)(x)

    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=96)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=96)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=96)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=160)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=160)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=160)(x)

    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=320)(x)

    x = Conv2D(1280, kernel_size=(1, 1))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)

    x = GlobalAveragePooling2D()(x)
    output = Dense(ClassNum, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model