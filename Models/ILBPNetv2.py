from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, ReLU,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D,AveragePooling2D,DepthwiseConv2D,LeakyReLU,add
from tensorflow.keras.layers import concatenate,Flatten,Dropout
from tensorflow.keras.regularizers import l2
from CNN_Module import make_divisible, inverted_res_block,correct_pad,ShuffleNetv2,fire_module,fire_module_ext

from tensorflow.keras import backend as K

def dirModule_ext(dir = 0 , sqz_filter = 32 , expand_filter = 64 ):
    def f(x,f,m):
        x = concatenate([x,f,m],axis=3)
        x = Conv2D(sqz_filter, (1, 1), padding='same')(x)
        x = LeakyReLU(0.3)(x)

        x = BatchNormalization()(x)

        x1 = Conv2D(expand_filter, (1, 3), padding='same')(x)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(expand_filter, (3, 1), padding='same')(x)
        x2 = Activation('relu')(x2)

        x = concatenate([x1, x2])
        
        return x
    return f
def dirModule(dir = 0 , sqz_filter = 32 , expand_filter = 64 ):
    def f(x):

        x = Conv2D(sqz_filter, (1, 1), padding='same')(x)
        x = LeakyReLU()(x)
        x1 = Conv2D(expand_filter, (1, 1), padding='same')(x)

        if dir == 0:    #ver extend
            x2 = DepthwiseConv2D((1,3))(x)
        else:           #hor extend
            x2 = DepthwiseConv2D((3,1))(x)
            
        x2 = Conv2D(expand_filter, (1, 1), padding='same')(x)
        x2 = Activation('relu')(x2)

        x = concatenate([x1, x2])            
        return x
    return f
def dirModulev2(dir = 0 , sqz_filter = 32 , expand_filter = 64 ):
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

    patial1 = partialModule(sqzFilter=32,outFilter=64,partialSize=3,isInterleave=True)(x)
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

def partialModule( sqzFilter , outFilter , partialSize ,sqzInc=0,leakInc=0, isInterleave = False ):
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

    #-------------------first pooling------------------------------------------------
    patial1 = partialModule(sqzFilter=32,outFilter=64,partialSize=3,isInterleave=True)(x)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(patial1)
    
    #
    f_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(f_mx_1)
    m_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(m_mx_1)
    mix = fire_module_ext(sqz_filter=32, expand_filter=64)(maxpool4,f_mx_2,m_mx_2)

    #-------------------second pooling-------------------------------------------
    d1 = dirModule(dir=0,sqz_filter=48, expand_filter=64)(mix)
    d2 = dirModule(dir=1,sqz_filter=48, expand_filter=64)(d1) 
    d5 = BusrtModule(2,sqzSize=8)(d2)
    d6 = BusrtModule(4,sqzSize=8)(d2)
    d7 = fire_module_ext(sqz_filter=96, expand_filter=128)(d2,d5,d6)
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(d7)


    #--------------------third pooling--------------------------------------
    f_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(f_mx_1)
    m_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(m_mx_1)
    mix_2= fire_module_ext(sqz_filter=96, expand_filter=128)(maxpool8,f_mx_sub,m_mx_sub)  


    d9 = BusrtModule(2,sqzSize=8)(mix_2)
    d10 = BusrtModule(4,sqzSize=8)(mix_2)
    mix_2= fire_module_ext(sqz_filter=72, expand_filter=128)(mix_2,d9,d10)


    d11 = BusrtModule(2,sqzSize=8)(mix_2)
    d12 = BusrtModule(4,sqzSize=8)(mix_2)
    fire9 = fire_module_ext(sqz_filter=128, expand_filter=256)(mix_2,d11,d12)

    fire9_dropout = Dropout(0.3)(fire9)

    return fire9_dropout

def DirNetv4(img,flbp,mlbp):
    i_cv_1 = Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same')(img)
    i_mx_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(i_cv_1)

    f_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(flbp)
    f_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(f_cv_1)

    m_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(mlbp)
    m_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(m_cv_1)

    x = concatenate([i_mx_1,f_mx_1,m_mx_1],axis=3)
    #49x49x18
    #----------------------基本資料產生---------------------------------------------------------
    FeatureGen1 =partialModule(sqzFilter=16,outFilter=64,sqzInc=0,partialSize=2,isInterleave=True)(x)
    partialExt1 = partialModule(sqzFilter=16,outFilter=64,sqzInc=0,partialSize=3,isInterleave=False)(FeatureGen1)#49x49x128 空間擴展
    FeatureGen2 = partialModule(sqzFilter=16,outFilter=64,sqzInc=0,partialSize=2,isInterleave=True)(partialExt1)#49x49x128 產生基本特徵

    #---------------------pooling--------------------------------------------------------------
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(FeatureGen2)    
    
    #---------------------feeding---------------------------------------------------------------
    f_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(f_mx_1)
    m_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(m_mx_1) 
    mix = concatenate([maxpool1,f_mx_2,m_mx_2])
    FeatureGen1 =partialModule(sqzFilter=20,outFilter=80,sqzInc=0,partialSize=2,isInterleave=True)(mix)
    partialExt1 = partialModule(sqzFilter=20,outFilter=80,sqzInc=0,partialSize=3,isInterleave=False)(FeatureGen1)#24x24x160
    FeatureGen2 = partialModule(sqzFilter=20,outFilter=80,sqzInc=0,partialSize=2,isInterleave=True)(partialExt1)#24x24x160

    #-------------------second pooling-------------------------------------------
    featureCom = partialModule(sqzFilter=16,outFilter=64,sqzInc=0,partialSize=1,isInterleave=False)(FeatureGen2)#49x49x64 空間擴展
    d5 = BusrtModule(2,sqzSize=8)(featureCom)#24x24x48
    d6 = BusrtModule(4,sqzSize=8)(featureCom)#24x24x48  //空間爆發
    #fedding
    d7 = fire_module_ext(sqz_filter=32, expand_filter=128)(featureCom,d5,d6)#24x24x256

    #降維
    maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(d7) #11x11x256
    #--------------------feeding--------------------------------------
    f_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(f_mx_1)
    m_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(m_mx_1)#11x11x1
    mix_2= fire_module_ext(sqz_filter=48, expand_filter=192)(maxpool2,f_mx_sub,m_mx_sub)  #11x11x256


    d9 = BusrtModule(2,sqzSize=4)(mix_2)#11x11x48
    d10 = BusrtModule(4,sqzSize=4)(mix_2)#11x11x48
    mix_2= fire_module_ext(sqz_filter=64, expand_filter=256)(mix_2,d9,d10)#11x11x256

    FeatureGen2 = partialModule(sqzFilter=64,outFilter=256,sqzInc=0,partialSize=1,isInterleave=True)(mix_2)#49x49x128 產生基本特徵


    #d11 = BusrtModule(2,sqzSize=8)(mix_2)#11x11x48
    #d12 = BusrtModule(4,sqzSize=8)(mix_2)#11x11x48
    #fire9 = fire_module_ext(sqz_filter=196, expand_filter=256)(mix_2,d11,d12)

    fire9_dropout = Dropout(0.3)(FeatureGen2)

    return fire9_dropout
def BusrtModule(BurstTimes,sqzSize = 8,expendSize = 32):
    def f(x):
        
        for i in range(BurstTimes):
            x = dirModule(dir=0,sqz_filter=sqzSize,expand_filter=expendSize)(x)
        for i in range(BurstTimes):
            x = dirModule(dir=1,sqz_filter=sqzSize*2,expand_filter=expendSize)(x)
        return x
    return f


def DirNetv5(img):
    '''i_cv_1 = Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same')(img)
    i_mx_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(i_cv_1)

    f_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(flbp)
    f_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(f_cv_1)

    m_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(mlbp)
    m_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(m_cv_1)'''

    #x = concatenate([i_mx_1,f_mx_1,m_mx_1],axis=3)

    #x = concatenate([img,flbp,mlbp],axis=3)
    conv1 = Conv2D(96, (7, 7), activation='relu', strides=(2, 2), padding='same')(img)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

    fire2 = fire_module(sqz_filter=16, expand_filter=64)(maxpool1)
    #burst = BusrtModule(4,sqzSize=4,expendSize=20)(fire2)
    #mix_1 = concatenate([fire2,burst],axis=3)
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
    

    return fire9_dropout



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
    x2 = DirNetv5(ori)

    x2 = Conv2D(ClassNum, (1, 1), activation='relu', padding='valid')(x2)

    x = GlobalAveragePooling2D()(x2)

    output = Dense(ClassNum, activation='softmax')(x)
    #output = Dense(ClassNum, activation='linear',kernel_regularizer=l2(0.001))(x)

    model = Model(inputs=input, outputs=output)

    return model