from keras.datasets import mnist
import keras
import torch
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

from PlasticNet import PlasticNet

num_classes = 10
epochs = 16
batch_size = 256

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255.0
x_test /= 255.0
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes).astype(np.float32)
y_test = keras.utils.to_categorical(y_test, num_classes).astype(np.float32)


plastic_net = PlasticNet(784, 50, 10, batch_size)
hebb = plastic_net.initial_zero_hebb()
hidden = plastic_net.initial_zero_state()

optimizer = Adam(plastic_net.parameters())
total_iterations = x_train.shape[0] // batch_size
loss = None
for _ in range(epochs):
    for i in range(total_iterations):
        current_x_train = x_train[i*batch_size:(i+1)*batch_size]
        current_y_train = y_train[i * batch_size:(i + 1) * batch_size]
        output, hidden, hebb = plastic_net(current_x_train, hidden, hebb)
        loss = -torch.mean(Variable(torch.from_numpy(current_y_train)) * torch.log(output))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    print(loss)

hebb = plastic_net.initial_zero_hebb()
hidden = plastic_net.initial_zero_state()
output = plastic_net(x_test[:batch_size], hidden, hebb)
print(output)
print(y_test)