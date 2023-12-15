
import tensorflow as tf
from tensorflow import keras
import os

import numpy as np


from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
import pandas as pd

import datetime

from init_model import create_MobileNet, create_DensNet



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # 
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # memory growth must be set before GPUs have been initialized
    print(e)



def init_train_validation(train_dir,validation_dir):
    # images rescaled by 1./255
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        width_shift_range = 32,
        height_shift_range = 32,
        shear_range=0.3,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        width_shift_range = 32,
        height_shift_range = 32,
        shear_range=0.3,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        seed = 0)
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator

def show_acc_loss(history,name,result_dir):
    
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
  
    
   
    
    
    plt.rcParams['axes.linewidth'] = 1 
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.grid(True)
    
    plt.set_yticks([0.4, 0.5, 1])
    plt.yticks(fontsize=10)
    ## 
    #plt.title(name+' Training and validation accuracy')
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'm', label='Validation acc')
    
    plt.legend()
    plt.savefig(result_dir+name+"acc")
    plt.figure()
    plt.clf()
    plt.close()
    
    
    
    
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    
    plt.set_yticks(np.arange(0.10, 1.10, 0.1))
    plt.yticks(fontsize=10)
        
    #plt.yticks(np.arange(0.10, 1.10, 0.1),fontsize=10)
    plt.grid(True)
    ##
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    #plt.title(name+' Training and validation loss')
    plt.legend()
    plt.savefig(result_dir+name+"loss")
    plt.figure()
    
    

def show_roc(model,name,result_dir,validation_dir):
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode='binary',
        shuffle=False)
    # calc threshold
    y_pred = model.predict(validation_generator).ravel()
    fpr, tpr, thresholds_keras = roc_curve(validation_generator.classes, y_pred)

    auc = metrics.auc(fpr, tpr)
    circle = np.zeros(len(fpr))
    youden = np.zeros(len(fpr))
    for i in range(len(fpr)):
        circle[i] = (1-tpr[i])*(1-tpr[i]) + fpr[i]*fpr[i]
        youden[i] = tpr[i] - fpr[i]

    circle_min_index = np.argmin(circle)
    print("circle: "+str(circle[circle_min_index]))
    print("threshhold: "+str(thresholds_keras[circle_min_index]))
    print("tpr,fpr:"+str(tpr[circle_min_index])+","+str(fpr[circle_min_index]))

    youden_max_index = np.argmax(youden)
    print("\nyouden: "+str(youden[youden_max_index]))
    print("threshhold: "+str(thresholds_keras[youden_max_index]))
    print("tpr,fpr:"+str(tpr[youden_max_index])+","+str(fpr[youden_max_index]))
    with open(result_dir+"threshhold.txt", mode='w') as f:
        f.write("auc:"+str(auc)+"\n"+
                "circle: "+str(circle[circle_min_index])+"\n" +
                "threshhold: "+str(thresholds_keras[circle_min_index])+"\n" +
                "tpr,fpr:"+str(tpr[circle_min_index])+","+str(fpr[circle_min_index])+"\n" +
                "\nyouden: "+str(youden[youden_max_index])+"\n" +
                "threshhold: "+str(thresholds_keras[youden_max_index])+"\n" +
                "tpr,fpr:"+str(tpr[youden_max_index])+","+str(fpr[youden_max_index]))
    plt.clf()
    
    
    plt.plot(fpr, tpr, 'b', label='ROC curve (AUC = %.2f)'%auc) #
    plt.legend()
    plt.xticks(np.arange(0, 1.1, 0.1),fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1),fontsize=15)
    plt.xlabel('False positive rate ',fontsize=13) # from 24
    plt.ylabel('True positive rate ',fontsize=13) # from 24 
    # plt.rcParams['axes.linewidth'] = 1 # 
    plt.rcParams['xtick.major.width'] = 1  #x軸の主目盛りの太さ
    plt.rcParams['ytick.major.width'] = 1  #y軸の主目盛りの太さ
    plt.rcParams['xtick.minor.width'] = 1  #x軸の補助目盛りの太さ
    plt.rcParams['ytick.minor.width'] = 1
    plt.grid(True)
    plt.savefig(result_dir+name+"roc")
    plt.show()  


def myprint(s):
    with open('model_summary.txt','a') as f:
        print(s, file=f)



# train ディレクトリ内の画像の枚数
num_train_data = 1200
# validation ディレクり内の画像の枚数
num_val_data = 400

img_size = 320
batch_size = 20
epochs = 150


date = datetime.date.today() #todays date
out_dir = 'result/'+str(date)+str(epochs+1)

#学習するモデルの指定
names = [
     "DenseNet",
     "MobileNet"
]

def main():
    
    train_dir = 'image/resized1024/train'
    validation_dir = 'image/resized1024/validation'
    train_generator, validation_generator = init_train_validation(train_dir,validation_dir)
    

    for i in range(1,3):
        result = out_dir+'_v'+str(i) + '/'
        os.mkdir(result)
        model = None
        
        for name in names:
            if name == "DenseNet":
                model, conv_base = create_DensNet(img_size)
                
            elif name == "MobileNet":
                model, conv_base = create_MobileNet(img_size)    
    
            else:
                break
            
            
                       
            history = model.fit(
                train_generator,
                steps_per_epoch=num_train_data / batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_val_data / batch_size,
                # seed = 0,
            )
            result_dir = os.path.join(result, name +"\\")
            os.mkdir(result_dir)
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(result_dir+'history.csv')
            model.save(result_dir+name+".h5")
            show_acc_loss(history,name,result_dir)
            show_roc(model, name,result_dir,validation_dir)
            
            model.summary(print_fn=myprint)
            
            model = None
          
    

if __name__ == '__main__':
    main()
