
# 圆盘分割算法代码使用 - U-Net in tensorflow

## Project info
这个代码使用于猪体长测量项目中圆盘的分割提取部分，能够很好的将圆盘从图片中分割出来

## Training
1、使用prepare.py用于生成训练样本的名称的csv文件，请修改该脚本第七行的path路径注意：  请务必保证训练数据
和label的名字相同；  
3、运行train.py用于开始训练；

## Inference
运行test.py来的得到测试结果

## Models
/model_v1_80 加入四川数据
/model_v2_50 增广数据集（旋转、Gaussian filtering） 训练过程中进行数据增强
/model_v3_50 修正u-net bug 使用depthwise_conv input_704x704 -batch_size: 16
/model_v4_20 调整损失函数: weighted_ce_loss(x5) 学习率衰减: exp_0.95_per_epoch
/model_v4_30 调整损失函数: iou_loss
/model_v5_50 input_768x768、训练数据加入负样本、损失函数为weighted_ce_loss和iou_loss之和
/model_v6_50 清洗、扩充数据集（加入莒南数据）
/model_v6_50 清洗、扩充数据集（加入泗水数据）
