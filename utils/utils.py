import random

import numpy as np
import torch
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:#判断是为RGB图像，np.shape(image) 用于获取图像数组的维度信息，返回一个元组（如 (高度, 宽度, 通道数)）。
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):#重设图像大小
    iw, ih  = image.size#原始图像的宽和高
    w, h    = size#目标图像的宽和高

    scale   = min(w/iw, h/ih)#计算缩放比例，选择较小的比例以保持图像的宽高比
    nw      = int(iw*scale)#新的宽度
    nh      = int(ih*scale)#新的高度

    image   = image.resize((nw,nh), Image.BICUBIC)#使用双三次插值法调整图像大小，保持图像质量
    #Image.BICUBIC 是 Python 图像处理库 Pillow（PIL） 中定义的一种双三次插值算法，主要用于图像缩放（放大 / 缩小）操作，以获得比简单插值更平滑、细节保留更好的图像效果。
    new_image = Image.new('RGB', size, (128,128,128))#创建一个新的灰色背景图像，大小为目标尺寸，填充颜色为灰色
    #(128, 128, 128) 是单个颜色值，其各通道的取值范围均为 0-255（这是 8 位 RGB 图像的标准范围，每个通道用 1 字节存储）
     
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))#将调整大小后的图像粘贴到新图像的中央位置

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):#获取优化器的学习率
    for param_group in optimizer.param_groups:#optimizer.param_groups 是一个包含多个参数组的列表，每个参数组都是一个字典，包含了该组参数的相关信息，如学习率、动量等。
        return param_group['lr']#返回第一个参数组的学习率

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):#归一化
    #preprocess预处理
    image /= 255.0
    return image

def show_config(**kwargs):#显示配置信息
    # kwargs 接受任意数量的关键字参数，并将它们作为字典传递给函数。在函数内部，可以通过访问 kwargs 字典来获取传递的参数及其对应的值。
    #config configuration 配置
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',#手动下载了一份放在./ManualDownload/vgg16-397923af.pth
        'resnet50'  : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'#手动下载了一份放在./ManualDownload/resnet50-19c8e357.pth
    }
    url = download_urls[backbone]
    if os.path.exists("./model_data/vgg16-397923af.pth") :
        if os.path.exists("./model_data/resnet50-19c8e357.pth"):
            print('file already exists.')
            return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)