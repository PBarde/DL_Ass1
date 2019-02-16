from __future__ import print_function
import sys
sys.path.insert(0, "./..")
import torch
from P3.main import distrib_means, distrib_stds, classes_dict
import torchvision.transforms
import os.path as osp
import csv

from torch.utils.data.sampler import SubsetRandomSampler

from_numpy = torch.from_numpy


class IdImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        """
         Args:
             index (int): Index

         Returns:
             tuple: (image, target) where target is class_index of the target class.
         """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        p, n = osp.split(path)
        n = n.split('.')[0]
        return img, target, n



if __name__=='__main__':

    p_best = './new_model_full_data_aug/d_aug_valid_acc_0.95.pth'
    model = torch.load(p_best)
    model.eval()

    batch_size = 64

    dogNcat_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(distrib_means, distrib_stds)])

    dogNcat_test = IdImageFolder(root='./testset',
                                 transform=dogNcat_transforms)

    test_loader = torch.utils.data.DataLoader(
        dogNcat_test, batch_size=batch_size, shuffle=False, num_workers=2)

    with open('submission_new_model.csv', mode='w') as csv_file:
        fieldnames = ['id', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for batch in test_loader:
            x, y, id = batch
            x = x.cuda()
            raw_labels = model(x).max(1)[1].cpu().numpy()
            for i, l in zip(id, raw_labels):
                writer.writerow({'id': i, 'label': classes_dict[l]})
