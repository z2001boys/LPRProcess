from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, ReLU,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D,AveragePooling2D,DepthwiseConv2D,LeakyReLU,add,GlobalMaxPooling1D
from tensorflow.keras.layers import concatenate,Flatten
from tensorflow.keras.regularizers import l2
from CNN_Module import make_divisible, inverted_res_block,correct_pad,ShuffleNetv2
import tensorflow.keras.backend as K
import numpy as np

def mobileV2(img):
    first_block_filters = make_divisible(32*1, 8)
    
    x = Conv2D(first_block_filters, kernel_size=(3, 3),
               strides=(2, 2), padding='valid')(img)
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

    return x

def specialConv(x):
    K.equal(x,1)

def smallNet(x):
    x = AveragePooling2D()(x)
    x = Conv2D(32,kernel_size=(3,3),strides=(1,1))(x) #extend to 128
    #48x48x32
    

    x = DepthwiseConv2D(kernel_size=(3,3),strides=(1,1))(x)
    x = Conv2D(64,1)(x)
    x = MaxPooling2D()(x)
    x = LeakyReLU()(x)
    #23x23x128

    
    x = DepthwiseConv2D(kernel_size=(3,3),strides=(2,2))(x)
    x = Conv2D(256,1)(x)
    x = BatchNormalization()(x)
    #11x11x128

    #do shuffle
    
    
    y1 = DepthwiseConv2D(kernel_size=(3,3),strides=(1,1))(x)
    y1 = ZeroPadding2D(padding=(1,1))(y1)
    x = concatenate([x, y1], axis=3)
    x = BatchNormalization()(x)
    #11x11x512

    x = Conv2D(1024,1)(x)
    x = DepthwiseConv2D(kernel_size=(3,3),strides=(1,1))(x)
    x = Conv2D(1280,1)(x)
    #9x9x768

    x = MaxPooling2D()(x)
    # 5x5x768
    

    return x

def reshapeImg(img):
    x = K.reshape(img,shape=(-1,100,100,1))
    return x

def gradientDistance(x):
    #remaping
    #x = x[]
    y = K.ones((10,10))
    t = K.set_value(x,y)
    
    return t



def GetMdl( inputShape ,ClassNum ):

    input = Input(shape=inputShape)


    ori = Lambda(lambda x: x[:,0:10000] ,output_shape=(-1,10000) )(input)
    imgData = Lambda(reshapeImg)(ori)
    
    firstSeleSize = 3240
    maxSeleSize = 2250
    MaxSelection = Lambda(lambda x: x[:,10000:10000+firstSeleSize] ,output_shape=(-1,firstSeleSize) )(input)
    FirstSelection = Lambda(lambda x: x[:,10000+firstSeleSize:10000+firstSeleSize+maxSeleSize] ,output_shape=(-1,maxSeleSize) )(input)
    
    #mobile net
    mbRow = mobileV2(imgData)
    mbFlatten = GlobalAveragePooling2D()(mbRow)
    mbFlatten = Lambda(lambda x:K.reshape(x,shape=(-1,1280,1)))(mbFlatten)
    
    combine1 = concatenate([mbFlatten,MaxSelection],axis=1)
    combine2 = concatenate([combine1,FirstSelection],axis=1)

    #x = BatchNormalization()(combine2)
    #x = Lambda(lambda x: K.reshape(x,shape=(-1,firstSeleSize+maxSeleSize+1280)))(x)
    #x = Dense(3000,activation='softmax')(x)
    y = Lambda(gradientDistance)(combine2)

    x = GlobalMaxPooling1D()(combine2)
    

    #orther cat methods

    output = Dense(ClassNum, activation='softmax')(x)

    #output = Dense(ClassNum, activation='linear',kernel_regularizer=l2(0.001))(x)

    model = Model(inputs=input, outputs=output)

    return model