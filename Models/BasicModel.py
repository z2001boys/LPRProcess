
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, ReLU,Flatten,Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from ILBPLayer import MyLayer
from CNN_Module import make_divisible, inverted_res_block,correct_pad

def GetMdl(inputShape,outputSize):
    input = Input(shape=inputShape)

    x = Conv2D(32, (3, 3), activation='relu')(input)# 100x100x32
    x = MaxPooling2D(pool_size=(2, 2))(input)   #33x33x32


    x = Conv2D(64, (3, 3), activation='relu')(x) #33x33x64
    x = MaxPooling2D(pool_size=(2, 2))(x) #16x16x64

    x = Conv2D(512, (5, 5), activation='relu')(x) #16x16x512
    x = MaxPooling2D(pool_size=(2, 2))(x)   # 8x8x512

    x = Conv2D(1024, (3, 3), activation='relu')(x) #16x16x512
    x = MaxPooling2D(pool_size=(2, 2))(x)   # 8x8x512


    x = Flatten()(x)    #32000
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)


    output = Dense(outputSize, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    
    return model