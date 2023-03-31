import os
import time
import uuid
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import numpy as np
from torch import cuda
from torch.utils.data import DataLoader
import cupy
import kvikio
from transformers import BertModel,BertTokenizer
#from cupy.core.dlpack import toDlpack
#from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from sys import getsizeof
import fastensorcpy
import fastensor_nodali

device = torch.device('cuda:0')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224))
])

# train_data = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
# test_data = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)

train_data = torchvision.datasets.Flowers102(root='./data',split='test',transform=transform,download=True)
test_data = torchvision.datasets.Flowers102(root='./data',split='train',transform=transform,download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("train_data_size:{}".format(train_data_size))
print("test_data_size:{}".format(test_data_size))

train_dataloader = DataLoader(train_data,batch_size=16)
test_dataloader = DataLoader(test_data,batch_size=16)

#hook
class SelfDeletingTempFile():
    def __init__(self):
        self.name = os.path.join("/home/lthpc/nvmessd/wj/tempfile",str(uuid.uuid4()))
        #self.name = os.path.join("./tempfile", str(uuid.uuid4()))
    #def __del__(self):
    #    os.remove(self.name)


''' numpy and cupy'''
def pack_hook(t):
    temp_file = SelfDeletingTempFile()
    file = fastensor_nodali.save(t, path = temp_file.name, policy='None',type='r')
    return file

def unpack_hook(file):
    t = fastensor_nodali.load(file)
    os.remove(file[0])
    return t

''' Network architecture '''
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()

        # self.model1=nn.Sequential(
        # nn.Conv2d(3, 32, 5, padding=2),
        # nn.MaxPool2d(2),
        # nn.Conv2d(32, 32, 5, padding=2),
        # nn.MaxPool2d(2),
        # nn.Conv2d(32, 64, 5, padding=2),
        # nn.MaxPool2d(2),
        # nn.Flatten(),  # 展平
        # nn.Linear(64 * 4 * 4, 64),
        # nn.Linear(64, 10),
        # )
        self.model1 = models.vit_h_14(num_classes = 102)

    def forward(self, x):  # input:32*32*3
        with torch.autograd.graph.saved_tensors_hooks(pack_hook,unpack_hook):
           x = self.model1(x)
        # x = self.model1(x)
        return x

net1=mymodel()
net1=net1.to(device)
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)
optimizer=torch.optim.SGD(net1.parameters(),lr=0.01)

total_train_step=0
total_test_step=0
epoch=20

print(next(net1.parameters()).device)
total = sum([param.nelement() for param in net1.parameters()])
print("Number of parameter: %.2f M"  % (total/1024/1024))
print("Memory of parameter: %.2f M " % (cuda.memory_allocated()/1024/1024))


'''start training'''
totalbegin=time.time()
for i in range(epoch):
    print("--------------The {} training begins----------".format(i+1))

    running_loss=0
    running_correct=0

    begin = time.time()
    flag = 0
    for data in train_dataloader:
        images,targets=data
        #print(images.device)
        images=images.to(device)
        targets=targets.to(device)

        outputs=net1(images)
        loss=loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        running_correct += (outputs.argmax(1) == targets).sum()
        total_train_step+=1
        #if total_train_step%100==0:
        #    print("number of training:{},loss:{}".format(total_train_step,loss))
        end2 = time.time()
        print("Batch ",flag," per batch spend time is", end2 - begin)
        flag += 1
    end = time.time()
    print("spend time: ",end - begin," s")
    print("epoch:{}, loss:{}, accuracy:{}".format(i+1,running_loss/train_data_size,running_correct/train_data_size))
    #break

totalend = time.time()
print("total real runtime: ", totalend - totalbegin, " s")

print("gpu memory allocated: %2.f M " % (cuda.memory_allocated()/1024/1024))



