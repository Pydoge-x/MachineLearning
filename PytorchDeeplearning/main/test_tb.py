from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")

image_path  = "Datasets/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)
img_arr = np.array(img)

print(img_arr.shape)

writer.add_image("test", img_arr, 2, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()