#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# In[2]:


input_size = 784
hidden_size = 500
num_class = 10
num_epochs  = 5
batch_size = 100
learning_rate = 0.01


# In[3]:


train_dataset = torchvision.datasets.FashionMNIST(root = './data',
                                    train=True,
                                    transform = transforms.ToTensor(),
                                    download = True)


# In[4]:


test_dataset = torchvision.datasets.FashionMNIST(root = './data',
                                                train=False,
                                                transform=transforms.ToTensor())


# In[5]:


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# In[6]:


total_step = len(train_loader)


# In[7]:


class My_CNN(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels = 1, out_channels = 6,kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.FC1 = nn.Linear(12*4*4,120)
        self.relu3 = nn.ReLU()
        self.FC2 = nn.Linear(120,60)
        self.relu4 = nn.ReLU()
        self.FC3 = nn.Linear(60,num_class)
        
        #forward pass
    def forward(self,x):
        out = self.Conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.Conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        #print(out.shape)
        out = out.reshape(-1, 12*4*4)
        #print(out.shape)
        out = self.FC1(out)
        out = self.relu3(out)
        out = self.FC2(out)
        out = self.relu4(out)
        out = self.FC3(out)
            # softmax is not required as we are using crossentropy
        return out


# In[8]:


model_cnn = My_CNN(10)


# In[9]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_cnn.parameters(),lr = learning_rate)


# In[10]:


for epoch in range(1):
    for i,(images,labels) in enumerate(train_loader):
        #images = images.reshape(-1,28,28)
        #images = images.to(device)
        #labels = labels.to(device)
        
        # forward step
        #print(images.shape)
        output = model_cnn(images)
        loss = criterion(output,labels)
        
        # Backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 ==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                 .format(epoch+1,num_epochs,i+1,total_step,loss.item()))
        


# In[11]:


output.shape


# In[12]:


# Save the model checkpoint
torch.save(model_cnn.state_dict(), 'model_cnn.ckpt')


# In[15]:


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        #images = images.to(device)
        #labels = labels.to(device)
        outputs = model_cnn(images)
        #print((predicted==labels).sum())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

