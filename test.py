import os
import numpy as np
import pandas as pd
from PIL import Image
import paddle.vision.transforms as T
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50

save_path = 'Fovea_Localization_Results.csv'
file_path = 'PLAM/PALM-Testing400-Images'
imgs_name = os.listdir(file_path)

tpsize = 256

model = nn.Sequential(
    resnet50(pretrained=False),
    nn.LeakyReLU(),
    nn.Linear(1000, 2)
)
params = paddle.load('final2.pdparams')
model.set_state_dict(params)
model.eval()

inf_transforms = T.Compose([
    T.Resize((tpsize, tpsize), interpolation='bicubic'),
    T.ToTensor()
])

pre_data = []
for img_name in imgs_name:
    data_path = os.path.join(file_path, img_name)
    data = np.asarray(Image.open(data_path).convert('RGB'))
    H, W, _ = data.shape
    data = inf_transforms(data)
    data = data.astype('float32').reshape([1, 3, tpsize, tpsize])
    pred = model(data)
    pre = [None] * 2
    # 还原坐标
    pre[0] = pred.numpy()[0][0] * W / tpsize
    pre[1] = pred.numpy()[0][1] * H / tpsize
    print(img_name, pre)
    pre_data.append([img_name, pre[0], pre[1]])

df = pd.DataFrame(pre_data, columns=['FileName', 'Fovea_X', 'Fovea_Y'])
df.sort_values(by="FileName",inplace=True,ascending=True)  #千万记得排序！
df.to_csv(save_path, index=None)
