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
training_set = pd.read_csv("MNIST/bindigit_trn.csv", header = None)
training_set = np.array(training_set, dtype = "int")

test_set = pd.read_csv("MNIST/bindigit_tst.csv", header = None)
test_set = np.array(test_set, dtype = "int")



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

#displayMNIST(test_set, 137) # test 0

#displayMNIST_alt(test_set[74, :]) # test 7



# 3 - Convert data sets into Torch tensors
trn_data = torch.cuda.FloatTensor(training_set)
tst_data = torch.cuda.FloatTensor(test_set)



# 4 - Constructing the undercomplete architecture
class autoenc(nn.Module):
    def __init__(self, ):
        super(autoenc, self).__init__() # inheritence
        self.full_connection0 = nn.Linear(784, 128) # encoding weights
        self.full_connection1 = nn.Linear(128, 784) # decoding weights
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
num_epochs = 200
length = len(trn_data)

for epoch in range(num_epochs):
    train_loss = 0
    score = 0.
    
    for num_data in range(length):
        input = Variable(trn_data[num_data]).cuda() #.cuda() - to move to GPU
        # === forward propagation ===
        output = model(input)
        loss = criterion(output, trn_data[num_data]) # loss between ŷ and y
        # === backward propagation ===
        loss.backward()
        # === calculating epoch loss ===
        train_loss += np.sqrt(loss.item())
        score += 1.
        
        optimizer.step()
    
    print('epoch: ' + str(epoch) + '   loss: ' + str(train_loss/score))


