# pylori-detection-DCNN

uses the following libraries efficiently: tensorflow.keras, matplotlib, sklearn, pandas.

init_model.py initializes each CNN models with respective hyper parameters.

train.py trains the fully connected part of DCNN, also changing the architecture of FC layer is possible.

CNN change is also possible.

train.py prints the training process and outputs it into 'history.csv' using pandas and saves the parameters in .h5 file.

also train.py calculates and plots the ROC curve (receiver operating characteristic curve) of the trained models.

mobnet plot.py and densnet plot.py both plots the accuracy and loss curve for both training and validation(evaluation).
