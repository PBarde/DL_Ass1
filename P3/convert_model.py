import torch
import sys
sys.path.insert(0, "./..")
from P3.archi import *

path = './models/0.938.pth'
model = torch.load(path)
model.eval()
torch.save(model.state_dict(), path + 'e')
