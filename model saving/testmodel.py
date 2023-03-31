# from torchstat import stat
import torch
import torchvision.models as models
import fastensor_model
import time
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import os

import sys
'''
import cupy
import kvikio
a = cupy.arange(20)
f = kvikio.CuFile("test-file", "w")
f.write(a)
f.close()
print(a)
d = cupy.empty_like(a)
print(d)
f1 = 17
f2 = 20
for i in range(10):
    with kvikio.CuFile("test-file", "r") as f:
        #future1 = f.pread(d[:5])#把读到的东西赋给前五个 01234
        #future2 = f.pread(d[5:])#把读到的东西赋给后五个 01234
        future3 = f.pread(d[f1:f2], file_offset=d[:f1].nbytes)#读出17,18,19;f2位置的数是不读的，设的时候设大
        #print(d)
        #future1.get()  # Wait for first read
        #print(d)
        #future2.get()  # Wait for second read
        #print(d)
        future3.get()
        f2 = f1
        f1 = f1 - 3
    print(d)
assert all(a == d)
'''
resnet152 = models.resnet152()
# resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg19 = models.vgg19()
resnext101 = models.resnext101_64x4d()
vitb32 = models.vit_b_32()
swinb = models.swin_b()
vith14 = models.vit_h_14()
mobilenetV3 = models.mobilenet_v3_small()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet_v2 = models.mobilenet_v2()
# mobilenet_v3_large = models.mobilenet_v3_large()
# mobilenet_v3_small = models.mobilenet_v3_small()
# resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()
# efficientnet_b0 = models.efficientnet_b0()
# efficientnet_b1 = models.efficientnet_b1()
# efficientnet_b2 = models.efficientnet_b2()
# efficientnet_b3 = models.efficientnet_b3()
# efficientnet_b4 = models.efficientnet_b4()
# efficientnet_b5 = models.efficientnet_b5()
# efficientnet_b6 = models.efficientnet_b6()
# efficientnet_b7 = models.efficientnet_b7()
# regnet_y_400mf = models.regnet_y_400mf()
# regnet_y_800mf = models.regnet_y_800mf()
# regnet_y_1_6gf = models.regnet_y_1_6gf()
# regnet_y_3_2gf = models.regnet_y_3_2gf()
# regnet_y_8gf = models.regnet_y_8gf()
# regnet_y_16gf = models.regnet_y_16gf()
# regnet_y_32gf = models.regnet_y_32gf()
# regnet_x_400mf = models.regnet_x_400mf()
# regnet_x_800mf = models.regnet_x_800mf()
# regnet_x_1_6gf = models.regnet_x_1_6gf()
# regnet_x_3_2gf = models.regnet_x_3_2gf()
# regnet_x_8gf = models.regnet_x_8gf()
# regnet_x_16gf = models.regnet_x_16gf()
# regnet_x_32gf = models.regnet_x_32gf()
# fasterrcnn_res= models.detection.fasterrcnn_resnet50_fpn()
# retina_rcnn = models.detection.retinanet_resnet50_fpn(pretrained=False)
# ssd_vgg16 = models.detection.ssd300_vgg16(pretrained=False)
# ssdlite330_mobilenetv3 = models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)
# fcn_res101 = models.segmentation.fcn_resnet101()
# deeplab_res101 = models.segmentation.deeplabv3_resnet101()
# lra_mobv3 = models.segmentation.lraspp_mobilenet_v3_large()

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.model1=nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5, padding=2),
        nn.MaxPool2d(2),
        nn.Flatten(),  # 展平
        nn.Linear(64 * 4 * 4, 64),
        nn.Linear(64, 10),
        )

    def forward(self, x):  # input:32*32*3
        # with torch.autograd.graph.saved_tensors_hooks(pack_hook,unpack_hook):
        #    x=self.model1(x)
        x = self.model1(x)
        return x

bert_path = "./bert_model"
class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=10):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
            out_pool = outputs[1]  # 池化后的输出 [bs, config.hidden_size]
            logit = self.fc(out_pool)  # [bs, classes]
        return logit


device = "cuda:1"
epochs = 100
total_time = 0
#net = Bert_Model(bert_path)
net = mymodel()
path1 = "./tempfile/bert"
'''
for i in range(epochs):
    a = format(i,'03d')
    mypath = path1 + a
    model = net.to(device)
    begin = time.time()
    fastensor_model.save(model.state_dict(), path=mypath, type = "w", policy = 'None')
    end = time.time()
    duration = end - begin
    print("epoch ", i ,"spend save time is", duration)
    if(int(i) != 0):
        total_time += duration
print("average save time is", total_time/(epochs-1))
'''
'''
print(model.state_dict().keys() == out.keys())
value1 = list(model.state_dict().values())
value2 = list(out.values())
for i in range(len(value2)):
    print(i)
    print("读入的：", value1[i])
    print("读出的：", value2[i])
    print(value1[i] == value2[i])
'''

#model = net.to(device)
#modelsize = sys.getsizeof(model.state_dict())
#modelsize = round(modelsize, 6)
path2 = "./tempfile"
path = os.listdir(path2)
for i in range(epochs):
    a = format(i,'03d')
    for i in path:
        mypath = os.path.join(path2,i)
        if(str(a) in mypath and len(mypath)<=22):
            print(mypath)
            begin = time.time()
            fastensor_model.load(mypath)
            end = time.time()
            duration = end - begin
            print("epoch ", i ,"spend load time is", duration)
            if(str(a) >= '009'):
                total_time += duration
print("average load time is", total_time/(epochs-1))
