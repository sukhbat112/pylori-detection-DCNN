# -*- coding: utf-8 -*-


from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers

# learning_rate = 0.00001

def myprint(s):
    with open('modelsummary_dens.txt','a') as f:
        print(s, file=f)

def myprint0(s):
    with open('modelsummary_mob.txt','a') as f:
        print(s, file=f)
        
        
def create_MobileNet(img_size,learning_rate = 0.00002):
    from tensorflow.keras.applications.mobilenet import MobileNet
    conv_base = MobileNet(input_shape=(img_size, img_size, 3),
                          weights='imagenet',
                            include_top=False,
                            pooling="avg",
                            #alpha=1.0, 
                            #depth_multiplier=1
                            )
    #conv_base.load_weights('mobilenet_1_0_224_tf.h5')
    
    conv_base.summary(print_fn=myprint0)
    
    conv_base.trainable = True
    model = models.Sequential()
    model.add(conv_base)
    #model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  #optimizer=optimizers.Adamax(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-10),
                  metrics=['acc'])

    return model, conv_base

def create_DensNet(img_size,learning_rate = 0.00001):
    from tensorflow.keras.applications.densenet import  DenseNet169
    conv_base = DenseNet169(weights='imagenet',
                          include_top=False,
                          pooling="avg",
                          input_shape=(img_size, img_size, 3))
    conv_base.trainable = True
    
    conv_base.summary(print_fn=myprint)
    
    model = models.Sequential()
    model.add(conv_base)
    #model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adamax(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None),
                  metrics=['acc'])

    return model, conv_base