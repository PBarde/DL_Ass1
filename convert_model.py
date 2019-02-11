import torch
from archi import *

path = './d_aug_test_acc_0.938.pth'
model = torch.load(path)
model.eval()
torch.save(model.state_dict(), path + 'e')
