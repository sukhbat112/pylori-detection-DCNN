# -*- coding: utf-8 -*-

# データ拡張の例を示すソースコード

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# initialising the ImageDataGenerator class.
# we will pass in the augmentation parameters in the constructor

datagen = ImageDataGenerator(
		rescale=1./255,
        rotation_range=180,
        width_shift_range = 32,
        height_shift_range = 32,
        shear_range=0.3,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',)
	

img = load_img('infected.png')
x = img_to_array(img)

# reshaping the input image
x = x.reshape((1, ) + x.shape)

# generating and saving 5 augmented samples
# using the above defined parameters.
i = 0
for batch in datagen.flow(x, batch_size = 1,
						save_to_dir ='preview',
						save_prefix ='infected', save_format ='png'):
	i += 1
	if i > 5:
		break
