# Autoencoder MNIST

# 0 - Importing Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt



# 1 - Import the MNIST data set
training_set0 = pd.read_csv("MNIST/bindigit_trn.csv", header = None)
training_set0 = np.array(training_set0, dtype = "int")

training_set1 = pd.read_csv("MNIST/targetdigit_trn.csv", header = None)
#training_set1 = np.array(training_set1, dtype = "int")

test_set0 = pd.read_csv("MNIST/bindigit_tst.csv", header = None)
test_set0 = np.array(test_set0, dtype = "int")

test_set1 = pd.read_csv("MNIST/targetdigit_tst.csv", header = None)
#test_set1 = np.array(test_set1, dtype = "int")



# 2 - Visualizing the MNIST images
def displayMNIST(data, index):
    display_number = data[index, :]
    display_number = display_number.reshape([28, 28])
    plt.imshow(display_number, cmap='gray')
    plt.show
    
def displayMNIST_alt(image):
    image = image.reshape([28,28])
    plt.imshow(image, cmap='gray')
    plt.show

#displayMNIST(training_set0, 137) # test 0
#displayMNIST(test_set0, 137) # test 0



# 3 - Convert data sets into Torch tensors
trn_data = torch.cuda.FloatTensor(training_set0)
tst_data = torch.cuda.FloatTensor(test_set0)



# 4 - Constructing the undercomplete architecture
class autoenc(nn.Module):
    def __init__(self, nodes = 100):
        super(autoenc, self).__init__() # inheritence
        self.full_connection0 = nn.Linear(784, nodes) # encoding weights
        self.full_connection1 = nn.Linear(nodes, 784) # decoding weights
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.full_connection0(x)) # input encoding
        x = self.full_connection1(x) # output decoding
        return x



# 5 - Initializing autoencoder, squared L2 norm, and optimization algorithm
model = autoenc().cuda() #.cuda() - to move to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
                          lr = 1e-3, weight_decay = 1/2)



# 6 - Training the undercomplete autoencoders
num_epochs = 500
batch_size = 32
length = int(len(trn_data) / batch_size)

# 6.1 - Training a model with 100 nodes
loss_epoch1 = []

for epoch in range(num_epochs):
    train_loss = 0
    score = 0. 
    
    
    for num_data in range(length - 2):
        #print(str(num_data) + "; length = " + str(length) + "   num_data = " + str(num_data))
        batch_ind = (batch_size * num_data)
        input = Variable(trn_data[batch_ind : batch_ind + batch_size]).cuda() #.cuda() - to move to GPU
        # === forward propagation ===
        #print("batch_ind = " + str(batch_ind) + "   batch_ind + batch_size = " + str(batch_ind + batch_size))
        output = model(input)
        loss = criterion(output, trn_data[batch_ind : batch_ind + batch_size]) # loss between ŷ and y
        # === backward propagation ===
        loss.backward()
        # === calculating epoch loss ===
        train_loss += np.sqrt(loss.item())
        score += 1. #<- add for average loss error instead of total
        optimizer.step()
    
    loss_calculated = train_loss/score
    print('epoch: ' + str(epoch + 1) + '   loss: ' + str(loss_calculated))
    loss_epoch1.append(loss_calculated)
    
    
# 6.2 - Training a model with 100 nodes
model2 = autoenc(nodes = 200).cuda() #.cuda() - to move to GPU
optimizer2 = optim.Adam(model2.parameters(),
                          lr = 1e-3, weight_decay = 1/2)
num_epochs = 500
batch_size = 32
length = int(len(trn_data) / batch_size)
loss_epoch2 = []

for epoch in range(num_epochs):
    train_loss = 0
    score = 0. 
    
    
    for num_data in range(length - 2):
        #print(str(num_data) + "; length = " + str(length) + "   num_data = " + str(num_data))
        batch_ind = (batch_size * num_data)
        input = Variable(trn_data[batch_ind : batch_ind + batch_size]).cuda() #.cuda() - to move to GPU
        # === forward propagation ===
        #print("batch_ind = " + str(batch_ind) + "   batch_ind + batch_size = " + str(batch_ind + batch_size))
        output = model2(input)
        loss = criterion(output, trn_data[batch_ind : batch_ind + batch_size]) # loss between ŷ and y
        # === backward propagation ===
        loss.backward()
        # === calculating epoch loss ===
        train_loss += np.sqrt(loss.item())
        score += 1. #<- add for average loss error instead of total
        optimizer2.step()
    
    loss_calculated = train_loss/score
    print('epoch: ' + str(epoch + 1) + '   loss: ' + str(loss_calculated))
    loss_epoch2.append(loss_calculated)


# 6.3 - Training a model with 400 nodes
model3 = autoenc(nodes = 400).cuda() #.cuda() - to move to GPU
optimizer3 = optim.Adam(model3.parameters(),
                          lr = 1e-3, weight_decay = 1/2)
num_epochs = 500
batch_size = 32
length = int(len(trn_data) / batch_size)
loss_epoch3 = []

for epoch in range(num_epochs):
    train_loss = 0
    score = 0. 
    
    
    for num_data in range(length - 2):
        #print(str(num_data) + "; length = " + str(length) + "   num_data = " + str(num_data))
        batch_ind = (batch_size * num_data)
        input = Variable(trn_data[batch_ind : batch_ind + batch_size]).cuda() #.cuda() - to move to GPU
        # === forward propagation ===
        #print("batch_ind = " + str(batch_ind) + "   batch_ind + batch_size = " + str(batch_ind + batch_size))
        output = model3(input)
        loss = criterion(output, trn_data[batch_ind : batch_ind + batch_size]) # loss between ŷ and y
        # === backward propagation ===
        loss.backward()
        # === calculating epoch loss ===
        train_loss += np.sqrt(loss.item())
        score += 1. #<- add for average loss error instead of total
        optimizer3.step()
    
    loss_calculated = train_loss/score
    print('epoch: ' + str(epoch + 1) + '   loss: ' + str(loss_calculated))
    loss_epoch3.append(loss_calculated)



# 7 - Plot total loss error as function of the epochs
plt.plot(loss_epoch1, label = "Squared L2; 100 Nodes")
plt.plot(loss_epoch2, label = "Squared L2; 200 Nodes")
plt.plot(loss_epoch3, label = "Squared L2; 400 Nodes")
plt.legend()
plt.show()



# 8 - Testing the undercomplete autoencoder

# 8.1 - Model with n = 100
test_loss = 0
s = 0.
length = len(tst_data)


for num_data in range (length):
    input = Variable(tst_data[num_data]).cuda()
    # === forward propagation ===
    output = model(input)
    loss = criterion(output, tst_data[num_data])
    # === calculating loss ===
    test_loss += np.sqrt(loss.item())
    s += 1.

loss_calculated = test_loss/s
print('n: 100 ; loss: ' + str(loss_calculated))


# 8.2 - Model with n = 200
test_loss = 0
s = 0.
length = len(tst_data)


for num_data in range (length):
    input = Variable(tst_data[num_data]).cuda()
    # === forward propagation ===
    output = model2(input)
    loss = criterion(output, tst_data[num_data])
    # === calculating loss ===
    test_loss += np.sqrt(loss.item())
    s += 1.

loss_calculated = test_loss/s
print('n: 200 ; loss: ' + str(loss_calculated))


# 8.3 - Model with n = 400
test_loss = 0
s = 0.
length = len(tst_data)


for num_data in range (length):
    input = Variable(tst_data[num_data]).cuda()
    # === forward propagation ===
    output = model3(input)
    loss = criterion(output, tst_data[num_data])
    # === calculating loss ===
    test_loss += np.sqrt(loss.item())
    s += 1.

loss_calculated = test_loss/s
print('n: 400 ; loss: ' + str(loss_calculated))
