import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

path = 'PLAM/PALM-Testing400-Images'
flrs = np.array(pd.read_csv('Fovea_Localization_Results.csv'))
for flr in flrs:
    img = np.array(Image.open(os.path.join(path, flr[0])))
    x, y = flr[1:]
    plt.figure()  # 创建新的图像窗口
    plt.imshow(img.astype('uint8'))
    plt.plot(x, y, 'or')
    plt.show()
