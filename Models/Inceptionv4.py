from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import stem_v4, inception_a_v4, reduction_a_v4, inception_b_v4, reduction_b_v4, inception_c_v4

def GetMdl(inputShape, ClassNum ):

    input = Input(shape=inputShape)

    x = stem_v4()(input)

    for i in range(4):
        x = inception_a_v4()(x)

    x = reduction_a_v4()(x)

    for j in range(7):
        x = inception_b_v4()(x)

    x = reduction_b_v4()(x)

    for k in range(3):
        x = inception_c_v4()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.8)(x)
    x = Flatten()(x)
    output = Dense(ClassNum, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model