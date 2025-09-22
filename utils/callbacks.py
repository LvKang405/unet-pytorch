import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir#日志保存位置
        self.val_loss_flag  = val_loss_flag# 是否需要记录验证集loss

        self.losses         = []#初始化损失列表（存储每个epoch的损失）
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)## 创建日志目录（若不存在）os.makedirs(self.log_dir, exist_ok=True)  # 建议加 exist_ok=True 避免重复创建报错
        self.writer     = SummaryWriter(self.log_dir)# 初始化TensorBoard写入器（用于在TensorBoard中查看损失）

        # 尝试将模型结构添加到TensorBoard（可选，可视化网络结构）
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])# 创建一个与输入形状匹配的虚拟输入张量# 生成一个虚拟输入（batch=2, channel=3, H=input_shape[0], W=input_shape[1]）# 这里的2表示批量大小，可以根据需要调整
            self.writer.add_graph(model, dummy_input)# 将模型结构添加到TensorBoard # 向TensorBoard添加模型图
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):# 记录每个epoch的损失值 调用时机：每个训练 epoch 结束后（通常在训练循环中调用），传入当前 epoch 的训练损失和验证损失。
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # 1. 将损失添加到列表（内存中记录）
        self.losses.append(loss) # 记录训练损失
        if self.val_loss_flag:
            self.val_loss.append(val_loss)# 记录验证损失（若开启）
        
         # 2. 将损失写入文本文件（持久化保存）
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss)) # 追加写入训练损失
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss)) # 追加写入验证损失
                f.write("\n")

        ## 3. 将损失写入TensorBoard（实时可视化）    
        self.writer.add_scalar('loss', loss, epoch) # 标签、值、epoch
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot() # 4. 生成损失曲线图


    #绘制损失曲线 核心是用 matplotlib 绘制训练 / 验证损失曲线，并支持 Savitzky-Golay 滤波平滑（让曲线更易读，减少波动干扰）：
    def loss_plot(self):
        iters = range(len(self.losses))# x 轴为迭代次数（epoch 数）

        plt.figure()# 创建新图像
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')# # 绘制原始训练损失（红色实线）
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')# 绘制原始验证损失（珊瑚色实线）
            
        try:
             # 滤波窗口大小：epoch数<25时用5，否则用15（窗口需为奇数）
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
             # 对损失曲线平滑（3阶多项式拟合）
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass # 若scipy未安装或数据量不足，跳过平滑

        plt.grid(True)# 显示网格
        plt.xlabel('Epoch')# x轴标签
        plt.ylabel('Loss')# y轴标签
        plt.legend(loc="upper right")# 图例位置

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))# 保存图像

         # 清理画布，避免内存泄漏
        plt.cla()# 清除当前轴
        plt.close("all")# 关闭所有画布


class EvalCallback():# 计算并记录mIoU 调用时机：每个训练 epoch 结束后（通常在训练循环中调用），传入当前 epoch 的模型（用于评估）。
#     语义分割任务中，mIoU（Mean Intersection over Union，平均交并比）是衡量模型性能的核心指标。该类负责：
# 定期评估：每 period 个 epoch（如每 5 个 epoch）自动评估一次；
# 预测生成：对验证集图像生成分割预测图；
# mIoU 计算：对比预测图与真值图（GT），计算 mIoU；
# 结果保存：将 mIoU 记录到文本文件和可视化图表。
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()# 继承父类初始化# 若父类有初始化，需调用（此处父类是object，可省略）
        
        self.net                = net               # 待评估的模型
        self.input_shape        = input_shape       # 模型输入尺寸（如 (512, 512)，H×W）
        self.num_classes        = num_classes       # 分割类别数（如VOC2007是21类）
        self.image_ids          = image_ids         # 验证集图像ID列表（通常来自数据集的txt文件）
        self.dataset_path       = dataset_path      # 数据集根路径（如 "./VOCdevkit/VOC2007"）
        self.log_dir            = log_dir           # 评估结果保存目录（与LossHistory共享）
        self.cuda               = cuda              # 是否用GPU评估（True/False）
        self.miou_out_path      = miou_out_path     # 临时保存预测图的目录
        self.eval_flag          = eval_flag         # 是否开启评估（True/False）
        self.period             = period            # 评估周期（每N个epoch评估一次）
        
        # 处理image_ids：提取纯ID（假设原始格式是 "xxx.jpg 1"，只取"xxx"）
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]

        # 初始化mIoU和epoch记录列表（初始值为0，对应epoch=0
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:  # 若开启评估，初始化mIoU文本文件
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):# 获取单张图像的预测结果 调用时机：在评估过程中，对每张验证集图像调用，生成对应的分割预测图。
        # 核心逻辑：输入原始图像，经过预处理→模型预测→后处理，生成与原始图像尺寸一致的分割预测图（每个像素对应一个类别）


        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#

        # 1. 图像预处理：转RGB（避免灰度图报错，语义分割通常用RGB输入）
        image       = cvtColor(image)  # 自定义函数：将图像转为RGB（如灰度图转3通道）
        orininal_h  = np.array(image).shape[0]  # 原始图像高度
        orininal_w  = np.array(image).shape[1]  # 原始图像宽度
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#

         # 2. 加灰条resize：保持宽高比，避免图像拉伸（语义分割关键，防止目标变形）
        # resize_image：自定义函数，将图像缩放到input_shape，空白处填灰条（如128）
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#

        # 3. 数据格式调整：适配PyTorch输入（batch, channel, H, W）
        # preprocess_input：自定义函数，图像归一化（如减均值、除标准差，如ImageNet的均值）    
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # 4. 转为张量，放GPU（若可用） # 4. 模型预测（关闭梯度计算，加速并节省内存）
        with torch.no_grad():
            images = torch.from_numpy(image_data)# numpy→tensor
            if self.cuda:
                images = images.cuda()# 移到GPU（若开启）
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#

            # 前向传播：获取模型输出（假设net输出为 (batch, num_classes, H, W)）
            pr = self.net(images)[0]# 获取模型预测结果（假设net返回一个元组，取第一个元素作为输出） # 取第一个样本（batch=1）
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#

            # 概率转换：softmax将输出转为类别概率（CHW→HWC，方便后续处理）
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()# CHW→HWC，转为numpy # 转CPU→numpy
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#

             # 5. 后处理：截取灰条部分（恢复到resize前的有效区域）
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # 高度方向：去掉上下灰条
             # 宽度方向：去掉左右灰条

            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#

            # 6. 恢复原始尺寸：将预测图resize回原始图像大小（双线性插值）
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#

              # 7. 确定类别：每个像素取概率最大的类别（argmax）
            pr = pr.argmax(axis=-1)# 每个像素点的类别索引（二维数组，H×W） # HWC→HW（每个像素值为类别ID）

    # 8. 转为图像格式（PIL Image），方便保存和后续处理 # 8. 转为图像格式（类别ID→uint8，便于保存为png）
        image = Image.fromarray(np.uint8(pr))
        return image
    
    # 每N个epoch评估一次mIoU 调用时机：在每个训练 epoch 结束后调用，传入当前 epoch 编号和模型（用于评估）。：epoch 结束后执行评估
    def on_epoch_end(self, epoch, model_eval):#调用时机：每个训练 epoch 结束后，判断是否达到评估周期（epoch % period == 0），若达到则执行评估。

         # 1. 检查是否需要评估（开启评估+达到周期）
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval  # 更新模型（确保用最新训练的模型评估）

             # 2. 准备路径：真值图（GT）目录和预测图目录
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")# 真值图目录（假设VOC2007数据集结构）# 真值图目录（VOC格式）
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')# 预测图目录（临时保存）# 预测图保存目录
            
             # 创建临时目录（保存预测图）
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            # 创建临时目录（保存预测图）
            # os.makedirs(self.miou_out_path, exist_ok=True)
            # os.makedirs(pred_dir, exist_ok=True)
            
            # 3. 生成预测图（对验证集每张图像预测并保存）
             # 遍历验证集图像ID，生成并保存预测图# 3. 批量生成预测图（遍历所有验证集图像）
            print("Get miou.")
            for image_id in tqdm(self.image_ids):# tqdm：显示进度条
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#

                # 读取原始图像（VOC格式：JPEGImages目录存原图）
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#

                # 生成预测图并保存（预测图命名与真值图一致，便于后续对比）
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
             # 4. 计算mIoU（调用外部函数compute_mIoU，需提前实现）           
            print("Calculate miou.")

             # compute_mIoU：输入GT目录、预测目录、图像ID、类别数，返回（mIoU, 各类IoU, ...）
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100# 计算平均IoU（转为百分比）

                # 5. 记录mIoU结果
        self.mious.append(temp_miou)  # 加入mIoU列表
        self.epoches.append(epoch)   # 加入epoch列表

        # 写入文本文件（持久化）
        with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
            f.write(str(temp_miou))
            f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            #  # 图表美化
            # plt.grid(True)
            # plt.xlabel('Epoch')
            # plt.ylabel('Miou')
            # plt.title('A Miou Curve')
            # plt.legend(loc="upper right")


             # 图表美化
            plt.grid(True)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('mIoU (%)', fontsize=12)
            plt.title('mIoU Curve', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)  # mIoU通常随epoch上升，图例放右下角
            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"), dpi=300, bbox_inches='tight')

            # plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")

             # 清理：删除临时预测图目录（避免占用空间）
            shutil.rmtree(self.miou_out_path)
