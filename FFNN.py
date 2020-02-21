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
learning_rate = 0.001


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


# Definition of the network
class FFNN_class(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.FC1 = nn.Linear(784,500)
        self.relu = nn.ReLU()
        self.FC2 = nn.Linear(500,num_class)
        
    def forward(self,x):
        out = self.FC1(x)
        out = self.relu(out)
        out = self.FC2(out)
        return out


# In[7]:


model = FFNN_class(num_class)


# In[8]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


# In[9]:


total_step = len(train_loader)


# In[10]:


for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28)
        labels = labels
        
        # forward step
        output = model(images)
        loss = criterion(output,labels)
        
        # Backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 ==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                 .format(epoch+1,num_epochs,i+1,total_step,loss.item()))
        


# In[11]:


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


# In[13]:


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        #print(images.shape)
        images = images.reshape(-1, 28*28)
        #print(images.shape)
        labels = labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (correct ==1):
            print(images.shape())

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# In[ ]:




