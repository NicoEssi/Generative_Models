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
    def __init__(self, ):
        super(autoenc, self).__init__() # inheritence
        self.full_connection0 = nn.Linear(784, 400) # encoding weights
        self.full_connection1 = nn.Linear(400, 784) # decoding weights
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.full_connection0(x)) # input encoding
        x = self.full_connection1(x) # output decoding
        return x



# 5 - Initializing autoencoder, squared L2 norm, and optimization algorithm
model = autoenc().cuda() #.cuda() - to move to GPU
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(),
                          lr = 1e-2, weight_decay = 1/2)



# 6 - Training the undercomplete autoencoder
num_epochs = 20
length = len(trn_data)
loss_epoch = []

for epoch in range(num_epochs):
    train_loss = 0
    #score = 0. 
    
    
    for num_data in range(length):
        input = Variable(trn_data[num_data]).cuda() #.cuda() - to move to GPU
        # === forward propagation ===
        output = model(input)
        loss = criterion(output, trn_data[num_data]) # loss between ŷ and y
        # === backward propagation ===
        loss.backward()
        # === calculating epoch loss ===
        train_loss += np.sqrt(loss.item())
        #score += 1. <- add for average loss error instead of total
        optimizer.step()
    
    #loss_calculated = train_loss/score
    print('epoch: ' + str(epoch + 1) + '   loss: ' + str(train_loss))
    loss_epoch.append(train_loss)
        
# With 200 epoch and 128 hidden nodes, loss : 0.17136.


    
# 7 - Plot total loss error as function of the epochs
plt.plot(loss_epoch, label = "Squared L2", )
plt.legend()
plt.show()



# 8 - Test the undercomplete autoencoder

