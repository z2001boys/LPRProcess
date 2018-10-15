from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import conv_bn_relu, dense_block, transition_block
def GetMdl( ClassNum ):

    input = Input(shape=(100, 100, 2))

    nb_dense_block = 4
    nb_layer = [6, 12, 32, 32]
    growth_rate = 32
    compression=0.5

    x = conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    for block_idx in range(nb_dense_block-1):
        x, nb_filter = dense_block(nb_layer[block_idx], nb_filter=64, growth_rate=growth_rate)(x)
        x = transition_block(nb_filter, compression)(x)
        nb_filter = int(nb_filter*compression)

    x, nb_filter = dense_block(nb_layer[-1], nb_filter, growth_rate)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(ClassNum)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)

    return model