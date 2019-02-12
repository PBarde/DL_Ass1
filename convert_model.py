import torch
from archi import *

path = './res_L2_correctLR_retrain/d_aug_test_acc_0.938.pth'
model = torch.load(path)
model.eval()
torch.save(model.state_dict(), path + 'e')
