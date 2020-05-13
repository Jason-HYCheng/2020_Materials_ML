import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn import preprocessing as P
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

## define functions(often use in AI network)
def mse(target_train, target_pred):
    from keras import backend as B
    return B.mean(B.square(target_pred - target_train), axis=-1)
def rmse(target_train, target_pred):
    from keras import backend as B
    return B.sqrt(B.mean(B.square(target_pred - target_train), axis=-1))
def r_square(target_train, target_pred):
    from keras import backend as B
    SS_res =  B.sum(B.square(target_train - target_pred)) 
    SS_tot = B.sum(B.square(target_train - B.mean(target_train))) 
    return (1 - SS_res/(SS_tot + B.epsilon()))
def r_square_loss(target_train, target_pred):
    from keras import backend as B
    SS_res =  B.sum(B.square(target_train - target_pred)) 
    SS_tot = B.sum(B.square(target_train - B.mean(target_train))) 
    return 1 - ( 1 - SS_res/(SS_tot + B.epsilon()))

# initialization
g = 9.8
datanum = 10000
trainnum = int (datanum * 0.9)
testnum = datanum - trainnum
np.random.seed(datanum)
input_data = np.zeros((datanum, 2), dtype = float)
output_data = np.zeros((datanum, 2), dtype = float)
output_pred = np.zeros((testnum, 2), dtype = float)
height = np.zeros((datanum, 1), dtype = float)
distance = np.zeros((datanum, 1), dtype = float)

# preparing dataset
velocity = np.linspace(0, 50, datanum)
angle = np.linspace(0, math.pi, datanum)
np.random.shuffle(velocity)
np.random.shuffle(angle)
for i in range(datanum):
    height[i] = ( velocity[i] * math.sin(angle[i]) )**2 / ( 2 * g )
    distance[i] = velocity[i]**2 * math.sin(2 * angle[i]) / g
input_data = np.hstack((velocity.reshape(-1,1), angle.reshape(-1,1)))
output_data = np.hstack((height, distance))

input_train = P.scale(input_data[:trainnum])
output_train = P.scale(output_data[:trainnum])
input_test = P.scale(input_data[trainnum:])
output_test = P.scale(output_data[trainnum:])

# input_train = input_data[:trainnum]
# output_train = output_data[:trainnum]
# input_test = input_data[trainnum:]
# output_test = output_data[trainnum:]

# building the network
model = Sequential()
model.add(Dense(units = 8, input_dim = 2, activation = 'relu'))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 2))
sgd = optimizers.Adam()
model.compile(loss = 'mean_squared_error', optimizer = sgd, \
              metrics=["accuracy", "mean_squared_error", rmse, r_square])

# training the network above
result = model.fit(input_train, output_train, epochs = 1000, batch_size = 150, \
                   validation_data = (input_test, output_test), verbose = 1)

output_pred = model.predict(input_test)
print(model.summary())
## calculate results
n_list = list(range(1, testnum+1))
h = output_test[:, 0]
h_pre = output_pred[:, 0]
d = output_test[:, 1]
d_pre = output_pred[:, 1]
h_r = (r2_score(h, h_pre))**0.5
h_loss = MSE(h_pre, h)
d_r = (r2_score(d, d_pre))**0.5
d_loss = MSE(d_pre, d)
           
## plot training curves & results
plt.plot(result.history['r_square'])
plt.plot(result.history['val_r_square'])
plt.title('R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(n_list, h, label = 'real height')
plt.plot(n_list, h_pre, label = 'predicted height')
plt.xlabel('No. of data', fontsize = 18)
plt.ylabel('maximum of height', fontsize = 18)
plt.title('Height Prediction', color='black', fontsize = 18)
plt.legend(loc='upper right')
plt.text(0, 15, 'height Loss=%.4f' % h_loss,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.text(0, 3.5, 'height correlation=%.4f' % h_r,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.show()

plt.plot(h, h_pre)
plt.xlabel('real height', fontsize = 18)
plt.ylabel('predicted height', fontsize = 18)
plt.title('Height Prediction', color='black', fontsize = 18)
plt.text(-0.5, 3.5, 'height Loss=%.4f' % h_loss,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.text(-0.5, 3, 'height correlation=%.4f' % h_r,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.show()

plt.plot(n_list, d, label = 'real distance')
plt.plot(n_list, d_pre, label = 'predicted distance')
plt.xlabel('No. of data', fontsize = 18)
plt.ylabel('maximum of distance', fontsize = 18)
plt.title('Distance Predicion', color='black', fontsize = 18)
plt.legend(loc='upper right')
plt.text(0, 15, 'distance Loss=%.4f' % d_loss,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.text(0, 3.5, 'distance correlation=%.4f' % d_r,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.show()

plt.plot(d, d_pre)
plt.xlabel('real distance', fontsize = 18)
plt.ylabel('predicted distance', fontsize = 18)
plt.title('Distance Prediction', color='black', fontsize = 18)
plt.text(-2.5, 3, 'height Loss=%.4f' % h_loss,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.text(-2.5, 2.5, 'height correlation=%.4f' % h_r,verticalalignment = 'top',horizontalalignment = 'left', fontdict={'size': 12, 'color':  'black'})
plt.show()