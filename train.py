import warnings
import os
import pandas as pd
import numpy as np
import paddle
import paddle.vision.transforms as T
from paddle.io import Dataset
from PIL import Image
from paddle.io import DataLoader
import paddle.nn.functional as F

import paddle
import paddle.nn as nn
from paddle.vision.models import resnet50

tpsize = 256
split = 0.9
batch_size = 16

warnings.filterwarnings("ignore")  # 拒绝烦人的警告信息


class PLAMDatas(Dataset):
    def __init__(self, data_path, class_path, mode='train', transforms=None, re_size=tpsize):
        super(PLAMDatas, self).__init__()
        self.data_path = data_path
        # 从 CSV 文件中读取数据
        self.name_label = pd.read_csv(class_path, usecols=[0, 1, 2]).values
        lens = len(self.name_label)
        if mode == 'train':
            self.name_label = self.name_label[:int(split * lens)]
        else:
            self.name_label = self.name_label[int(split * lens):]
        self.transforms = transforms
        self.re_size = re_size

    def __getitem__(self, index):
        name, x, y = self.name_label[index]  # 得到的数据赋值一下
        data_path = os.path.join(self.data_path, f'{name}.jpg')
        data = np.asarray(Image.open(data_path).convert('RGB'))
        H, W, _ = data.shape
        if self.transforms is not None:
            data = self.transforms(data)
        data = data.astype('float32')

        label = np.array([x * self.re_size / W, y * self.re_size / H]).astype('float32')  # 图片大小变了，对应的坐标自然也要改变
        return data, label

    def __len__(self):
        return len(self.name_label)


# 配置数据增广
train_transforms = T.Compose([
    T.Resize((tpsize, tpsize), interpolation='bicubic'),  # 都调整到1800 选用bicubic，放大不至于太失真
    T.ToTensor()
])

val_transforms = T.Compose([
    T.Resize((tpsize, tpsize), interpolation='bicubic'),
    T.ToTensor()
])


# images_path = 'PLAM/Train/fundus_image'
# class_path = 'PLAM/Train/Fovea_Location_train.xlsx'

images_path = 'ds2/Training a and b-20180122T080847Z-001/Training a and b'
class_path = 'ds2/IDRiD_OD_Center_Training_set.csv'
# 配置数据集
train_dataset = PLAMDatas(data_path=images_path, class_path=class_path,
                          mode='train', transforms=train_transforms)
val_dataset = PLAMDatas(data_path=images_path, class_path=class_path,
                        mode='test', transforms=val_transforms)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True, drop_last=False)
dev_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True, drop_last=False)

print(len(train_dataset), len(val_dataset))
print(len(train_dataloader), len(dev_dataloader))

# 模型定义
model = nn.Sequential(
    resnet50(pretrained=True),
    nn.LeakyReLU(),
    nn.Linear(1000, 2)  # 坐标定位
)
paddle.summary(model, (1, 3, tpsize, tpsize))
model.set_state_dict(paddle.load('final2.pdparams'))
model = paddle.Model(model)



class FocusBCELoss(nn.Layer):
    '''
        本赛题的任务损失函数
    '''

    def __init__(self, weights=[0.5, 0.5]):
        super(FocusBCELoss, self).__init__()
        self.weights = weights  # 损失权重

    def forward(self, predict, label):
        # MSE均方误差
        mse_loss_x = paddle.nn.functional.mse_loss(predict[:, 0], label[:, 0], reduction='mean')
        mse_loss_y = paddle.nn.functional.mse_loss(predict[:, 1], label[:, 1], reduction='mean')
        mse_loss = 0.5 * mse_loss_x + 0.5 * mse_loss_y

        # 欧氏距离
        distance_loss = paddle.subtract(predict, label)
        distance_loss = paddle.square(distance_loss)
        distance_loss = paddle.sum(distance_loss, axis=-1)
        distance_loss = paddle.sqrt(distance_loss)
        distance_loss = paddle.sum(distance_loss, axis=0) / predict.shape[0]  # predict.shape[0] == batch_size

        alpha1, alpha2 = self.weights
        all_loss = alpha1 * mse_loss + alpha2 * distance_loss

        return all_loss, mse_loss, distance_loss


# 模型准备
epochs = 50

lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=2e-3, decay_steps=int(800*tpsize))
opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(5e-6))
loss  = FocusBCELoss(weights=[0.4, 0.6])   # weights，不同类别的损失权重

model.prepare(
    optimizer = opt,
    loss = loss
    )
visualdl=paddle.callbacks.VisualDL(log_dir='visual_log')

use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
print('Train on {} GPU'.format(paddle.get_device()))


# 模型微调
model.fit(
    train_data=train_dataset,
    eval_data=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    eval_freq=10,
    log_freq=1,
    save_dir='resnet_50_save_models_256_0.9_16',
    save_freq=10,
    verbose=1,
    drop_last=False,
    shuffle=True,
    num_workers=0,
    callbacks=[visualdl]
)

# save pdparams
# model.save('resnet_50_save_models_256_0.9_16/final')
model.save('resnet_50_save_models_256_0.9_16/final2')
