from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from CNN_Module import fire_module
def GetMdl( inputShape ,ClassNum ):

    input = Input(shape=inputShape)

    conv1 = Conv2D(96, (7, 7), activation='relu', strides=(2, 2), padding='same')(input)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

    fire2 = fire_module(sqz_filter=16, expand_filter=64)(maxpool1)
    fire3 = fire_module(sqz_filter=16, expand_filter=64)(fire2)
    fire4 = fire_module(sqz_filter=32, expand_filter=128)(fire3)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire4)

    fire5 = fire_module(sqz_filter=32, expand_filter=128)(maxpool4)
    fire6 = fire_module(sqz_filter=48, expand_filter=192)(fire5)
    fire7 = fire_module(sqz_filter=48, expand_filter=192)(fire6)
    fire8 = fire_module(sqz_filter=64, expand_filter=256)(fire7)
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire8)

    fire9 = fire_module(sqz_filter=64, expand_filter=256)(maxpool8)
    fire9_dropout = Dropout(0.5)(fire9)
    conv10 = Conv2D(ClassNum, (1, 1), activation='relu', padding='valid')(fire9_dropout)
    global_avgpool10 = GlobalAveragePooling2D()(conv10)
    output = Activation("softmax")(global_avgpool10)
    
    model = Model(inputs=input, outputs=output)

    return model