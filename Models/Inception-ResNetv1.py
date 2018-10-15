from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from ILBPLayer import MyLayer
from CNN_Module import stem_inception_resnet_v1, inception_resnet_a_v1, reduction_resnet_a, inception_resnet_b_v1, reduction_resnet_b_v1, inception_resnet_c_v1

def GetMdl( ClassNum ):

    input = Input(shape=(100, 100, 2))

    x = stem_inception_resnet_v1()(input)

    for i in range(5):
        x = inception_resnet_a_v1()(x)

    x = reduction_resnet_a(k = 192, l = 192, m = 256, n = 384)(x)

    for j in range(10):
        x = inception_resnet_b_v1()(x)

    x = reduction_resnet_b_v1()(x)

    for k in range(5):
        x = inception_resnet_c_v1()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    output = Dense(ClassNum, activation = "softmax")(x)

    model = Model(inputs=input, outputs=output)

    return model