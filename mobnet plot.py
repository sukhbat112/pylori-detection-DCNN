import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



input_csv = pd.read_csv('result/224_MobOnly_2023-01-20_v2/MobileNet/history.csv')



epochs = input_csv[input_csv.keys()[0]]
loss = input_csv[input_csv.keys()[1]]
acc = input_csv[input_csv.keys()[2]]
val_loss = input_csv[input_csv.keys()[3]]
val_acc = input_csv[input_csv.keys()[4]]

plt.rcParams['axes.linewidth'] = 1
plt.grid(True)
plt.plot(epochs, acc, color='g',markersize=4 ,marker="o", label='Training acc')
plt.plot(epochs, val_acc, color='m', label='Validation acc')
plt.title('mobnet accuracy')
#plt.xlim((0,50))

plt.yticks(np.arange(0.4, 1.1, 0.1),fontsize=10)
plt.legend()


plt.savefig('mob_acc.png')
plt.figure()

plt.clf



plt.rcParams['axes.linewidth'] = 1
plt.grid(True)
plt.plot(epochs, loss, color='g', markersize=4 ,marker="o", label='Training loss')
plt.plot(epochs, val_loss, color='m', label='Validation loss')
plt.title('mobnet loss')
#plt.xlim((0,50))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yticks(np.arange(0.10, 1.1, 0.1),fontsize=10)
plt.legend()

plt.savefig('mob_loss.png')

plt.show()