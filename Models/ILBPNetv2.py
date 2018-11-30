from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, ReLU,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D,AveragePooling2D,DepthwiseConv2D,LeakyReLU,add
from tensorflow.keras.layers import concatenate,Flatten,Dropout
from tensorflow.keras.regularizers import l2
from CNN_Module import make_divisible, inverted_res_block,correct_pad,ShuffleNetv2,fire_module,fire_module_ext

from tensorflow.keras import backend as K

def dirModule_ext(dir = 0 , sqz_filter = 32 , expand_filter = 64 ):
    def f(x,f,m):

        x = Conv2D(sqz_filter, (1, 1), padding='same')(x)
        x = LeakyReLU(0.3)(x)
        x = concatenate([x,f,m],axis=3)
        x = BatchNormalization()(x)

        if dir == 0:    #ver extend
            x1 = Conv2D(expand_filter, (1, 1), padding='same')(x)
            x1 = Activation('relu')(x1)
            x2 = Conv2D(expand_filter, (1, 3), padding='same')(x)
            x2 = Activation('relu')(x1)
        else:           #hor extend
            x1 = Conv2D(expand_filter, (1, 1), padding='same')(x)
            x1 = Activation('relu')(x1)
            x2 = Conv2D(expand_filter, (3, 1), padding='same')(x)
            x2 = Activation('relu')(x1)

            x = concatenate([x1, x2])            
        return x
    return f
def dirModule(dir = 0 , sqz_filter = 32 , expand_filter = 64 ):
    def f(x):

        x = Conv2D(sqz_filter, (1, 1), padding='same')(x)
        x = LeakyReLU()(x)

        if dir == 0:    #ver extend
            x1 = Conv2D(expand_filter, (1, 1), padding='same')(x)
            x1 = Activation('relu')(x1)
            x2 = Conv2D(expand_filter, (1, 3), padding='same')(x)
            x2 = Activation('relu')(x1)
        else:           #hor extend
            x1 = Conv2D(expand_filter, (1, 1), padding='same')(x)
            x1 = Activation('relu')(x1)
            x2 = Conv2D(expand_filter, (3, 1), padding='same')(x)
            x2 = Activation('relu')(x1)

            x = concatenate([x1, x2])            
        return x
    return f

def DirNet(img,flbp,mlbp):
    i_cv_1 = Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same')(img)
    i_mx_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(i_cv_1)

    f_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(flbp)
    f_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(f_cv_1)

    m_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(mlbp)
    m_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(m_cv_1)

    x = concatenate([i_mx_1,f_mx_1,m_mx_1],axis=3)
    #49x49x18

    patial1 = patialModule(sqzFilter=32,outFilter=64,partialSize=3,isInterleave=True)(x)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(patial1)

    f_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(f_mx_1)
    m_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(m_mx_1)
    mix = fire_module_ext(sqz_filter=32, expand_filter=128)(maxpool4,f_mx_2,m_mx_2)

    #ver
    d1 = dirModule(dir=0,sqz_filter=64, expand_filter=64)(mix)
    d2 = dirModule(dir=1,sqz_filter=64, expand_filter=64)(d1)
    d3 = dirModule(dir=0,sqz_filter=64, expand_filter=128)(d2)
    d4 = dirModule(dir=1,sqz_filter=64, expand_filter=128)(d3)    
    d5 = dirModule(dir=1,sqz_filter=64, expand_filter=128)(d4)
    d6 = dirModule(dir=1,sqz_filter=64, expand_filter=128)(d5)
    d7 = dirModule(dir=0,sqz_filter=72, expand_filter=256)(d6)
    d8 = dirModule(dir=0,sqz_filter=96, expand_filter=256)(d7)        
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(d8)

    f_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(f_mx_1)
    m_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(m_mx_1)
    mix_2= fire_module_ext(sqz_filter=96, expand_filter=128)(maxpool8,f_mx_sub,m_mx_sub)   

    fire9 = fire_module(sqz_filter=128, expand_filter=256)(mix_2)
    fire9_dropout = Dropout(0.2)(fire9)


    return fire9_dropout

def patialModule( sqzFilter , outFilter , partialSize ,sqzInc=0,leakInc=0, isInterleave = False ):
    def f( x ):
        curSqzSize = sqzFilter
        curOutSize = outFilter
        for i  in range(partialSize*2):
            dir = 0 
            if isInterleave == True:
                if i%2==1:
                    dir = 1
            else:
                if i >= partialSize:
                    dir = 1               
                    
            x = dirModule(dir=dir, sqz_filter=curSqzSize, expand_filter=curOutSize)(x)
            curSqzSize = curSqzSize + sqzInc
            curOutSize = curOutSize + leakInc

        return x
    return f

def DirNetv2(img,flbp,mlbp):
    i_cv_1 = Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same')(img)
    i_mx_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(i_cv_1)

    f_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(flbp)
    f_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(f_cv_1)

    m_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(mlbp)
    m_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(m_cv_1)

    x = concatenate([i_mx_1,f_mx_1,m_mx_1],axis=3)
    #49x49x18

    patial1 = patialModule(sqzFilter=32,outFilter=64,partialSize=3,isInterleave=True)(x)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(patial1)

    f_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(f_mx_1)
    m_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(m_mx_1)
    mix = fire_module_ext(sqz_filter=32, expand_filter=128)(maxpool4,f_mx_2,m_mx_2)

    #ver
    d1 = dirModule(dir=0,sqz_filter=64, expand_filter=64)(mix)
    d2 = dirModule(dir=1,sqz_filter=64, expand_filter=64)(d1)
    d3 = dirModule(dir=0,sqz_filter=64, expand_filter=128)(d2)
    d4 = dirModule(dir=1,sqz_filter=64, expand_filter=128)(d3)    
    d5 = dirModule(dir=1,sqz_filter=64, expand_filter=128)(d4)
    d6 = dirModule(dir=1,sqz_filter=64, expand_filter=128)(d5)
    d7 = dirModule(dir=0,sqz_filter=72, expand_filter=256)(d6)
    d8 = dirModule(dir=0,sqz_filter=96, expand_filter=256)(d7)        
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(d8)

    f_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(f_mx_1)
    m_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(m_mx_1)
    mix_2= fire_module_ext(sqz_filter=96, expand_filter=128)(maxpool8,f_mx_sub,m_mx_sub)   

    d9 = dirModule(dir=1,sqz_filter=128, expand_filter=128)(mix_2)    
    d10 = dirModule(dir=1,sqz_filter=128, expand_filter=128)(d9)
    d11 = dirModule(dir=1,sqz_filter=128, expand_filter=128)(d10)
    d12 = dirModule(dir=0,sqz_filter=128, expand_filter=256)(d11)
    d13 = dirModule(dir=0,sqz_filter=128, expand_filter=256)(d12) 
    
    fire9_dropout = Dropout(0.2)(d13)

    return fire9_dropout


def smallNet(x):
    x = AveragePooling2D()(x)

    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = inverted_res_block(t=1, strides=(2, 2), alpha=1, filters=32, is_expansion=False)(x)    
    #48x48x32
    

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=32, is_expansion=False)(x)
    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=64, is_expansion=False)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=128, is_expansion=False)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=128, is_expansion=False)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=256, is_expansion=False)(x)
    x = inverted_res_block(t=6, strides=(1, 1), alpha=1, filters=256, is_expansion=False)(x)

    x = inverted_res_block(t=6, strides=(2, 2), alpha=1, filters=256, is_expansion=False)(x)

    x = Conv2D(512, kernel_size=(1, 1))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    

    return x




def GetMdl( inputShape ,ClassNum ):

    input = Input(shape=inputShape)


    #ori = Lambda(lambda x: x[:,:,:,2] ,output_shape=(-1,100,100,2) )(input)
    #oriExt = Lambda(lambda x: K.expand_dims(x) ,output_shape=(-1,100,100,1) )(ori)

    flbp = Lambda(lambda x: x[:,:,:,0] ,output_shape=(-1,100,100,1) )(input)
    flbp = Lambda(lambda x: K.expand_dims(x) ,output_shape=(-1,100,100,1),name = 'split_channel' )(flbp)
    #flbp = Lambda(lambda x: x/9 ,output_shape=(-1,100,100,1),name='FLBP_Norm' )(flbp)

    mlbp = Lambda(lambda x: x[:,:,:,0] ,output_shape=(-1,100,100,1) )(input)
    mlbp = Lambda(lambda x: K.expand_dims(x) ,output_shape=(-1,100,100,1) )(mlbp)

    ori = Lambda(lambda x: x[:,:,:,2] ,output_shape=(-1,100,100,1) )(input)
    ori = Lambda(lambda x: K.expand_dims(x) ,output_shape=(-1,100,100,1) )(ori)

    #y = Lambda(specialConv)(oriExt)
    

    #x1 = smallNet(flbpExt)
    x2 = DirNetv2(ori,flbp,mlbp)

    x2 = Conv2D(ClassNum, (1, 1), activation='relu', padding='valid')(x2)

    x = GlobalAveragePooling2D()(x2)

    output = Dense(ClassNum, activation='softmax')(x)
    #output = Dense(ClassNum, activation='linear',kernel_regularizer=l2(0.001))(x)

    model = Model(inputs=input, outputs=output)

    return model