from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, ReLU,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D,AveragePooling2D,DepthwiseConv2D,LeakyReLU,add
from tensorflow.keras.layers import concatenate,Flatten,Dropout
from tensorflow.keras.regularizers import l2
from CNN_Module import make_divisible, inverted_res_block,correct_pad,ShuffleNetv2,fire_module,fire_module_ext

from tensorflow.keras import backend as K


def LimitNet(img,flbp,mlbp):
    i_cv_1 = Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same')(img)
    i_mx_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(i_cv_1)

    f_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(flbp)
    f_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(f_cv_1)

    m_cv_1 = Conv2D(1, (3, 3), activation='relu', strides=(1, 1), padding='same')(mlbp)
    m_mx_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(m_cv_1)

    x = concatenate([i_mx_1,f_mx_1,m_mx_1],axis=3)

    fire2 = fire_module(sqz_filter=32, expand_filter=64)(x)
    fire3 = fire_module(sqz_filter=64, expand_filter=64)(fire2)
    fire4 = fire_module(sqz_filter=64, expand_filter=128)(fire3)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire4)

    #second combine module
    f_mx_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(f_mx_1)
    f_mx_1_2 = Conv2D(1, (3, 3), activation='relu', strides=(2, 2))(f_mx_1)
    mix = fire_module_ext(sqz_filter=32, expand_filter=128)(maxpool4,f_mx_2,f_mx_1_2)
    

    fire5 = fire_module(sqz_filter=32, expand_filter=128)(mix)
    fire6 = fire_module(sqz_filter=48, expand_filter=192)(fire5)
    fire7 = fire_module(sqz_filter=48, expand_filter=192)(fire6)
    fire8 = fire_module(sqz_filter=64, expand_filter=256)(fire7)
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire8)


    #
    f_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(f_mx_1)
    m_mx_sub = AveragePooling2D(pool_size=(6, 6),strides=(4,4))(m_mx_1)
    mix_2= fire_module_ext(sqz_filter=64, expand_filter=128)(maxpool8,f_mx_sub,m_mx_sub)    



    fire9 = fire_module(sqz_filter=128, expand_filter=256)(mix_2)
    fire9_dropout = Dropout(0.2)(fire9)

    #fire10 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire9_dropout)
    #f_mx_3 = Conv2D(2, (1, 1), activation='relu', strides=(1, 1))(f_mx_2)
    #f_mp_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(f_mx_3)

    #m_mx_3 = Conv2D(2, (1, 1), activation='relu', strides=(1, 1))(m_mx_2)
    #m_mp_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(m_mx_3)
    
    #mix_3 = fire_module_ext(sqz_filter=192, expand_filter=512)(fire10,f_mp_3,m_mp_3)
    #mix_3 = Dropout(0.2)(mix_3)

    #mix_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mix_3)
    
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
    x2 = LimitNet(ori,flbp,mlbp)

    x2 = Conv2D(ClassNum, (1, 1), activation='relu', padding='valid')(x2)

    x = GlobalAveragePooling2D()(x2)

    output = Dense(ClassNum, activation='softmax')(x)
    #output = Dense(ClassNum, activation='linear',kernel_regularizer=l2(0.001))(x)

    model = Model(inputs=input, outputs=output)

    return model