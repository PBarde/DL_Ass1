import torch
from archi import *

path = './new_model_full_data_aug/d_aug_valid_acc_0.95.pth'
model = torch.load(path)
model.eval()
torch.save(model.state_dict(), path + 'e')
