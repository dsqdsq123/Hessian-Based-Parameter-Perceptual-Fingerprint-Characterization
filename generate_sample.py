import random
import torch
import models.models as clmgnet
from uilts import get_dataset, myway, select_sample_tensor

model = clmgnet.CLMGNet(num_classes=16, n_bands=200, ps=13, inplanes=256, num_blocks=4, num_heads=4, num_encoders=1)
model_state_dict = torch.load('./targetmodel.pth')
model.load_state_dict(model_state_dict['state_dict'])
img, gt, ignored_labels = get_dataset("./data/Indian_pines_corrected.mat", "./data/Indian_pines_gt.mat")
xmin = img.min()
xmax = img.max()
device = torch.device('cuda')
model = model.to(device)

i = 0  # You want to generate the coordinates of sensitive samples.
j = 0
k = 5  # The patch size of training samples.
x = img[i:i + k, j:j + k, :]
gt_centor = gt[i + k / 2, j + k / 2]  # gt of center
gt_centor = torch.tensor(gt_centor).unsqueeze(0).to(device)
lr = 0.0001
epoches = 10000
myway(x, model, lr, epoches, gt_centor, f"./sample_sen_entropy/({i},{j},{gt_centor}).pth",
      f"./sample_sen/({i},{j},{gt_centor}).pth", 0.5, 1.0, xmin,
      xmax)

# select_sample_tensor(model, "./sample_sen_entropy")
