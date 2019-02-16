from __future__ import print_function
import sys
sys.path.insert(0, "./..")
import torch
from P3.main import distrib_means, distrib_stds, classes_dict
import torchvision.transforms
import os.path as osp
import csv
from P3.archi import *

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

    submission_type = 'simple'
    batch_size = 64

    dogNcat_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(distrib_means, distrib_stds)])

    dogNcat_test = IdImageFolder(root='./testset',
                                 transform=dogNcat_transforms)

    test_loader = torch.utils.data.DataLoader(
        dogNcat_test, batch_size=batch_size, shuffle=False, num_workers=2)

    if submission_type == 'simple':
        p_best = '.models/0.938.pthe'
        model = CIFARResNet18(num_classes=2, k=3).cuda()
        model.load_state_dict(torch.load(p_best))
        model.eval()

        with open('submission.csv', mode='w') as csv_file:
            fieldnames = ['id', 'label']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for batch in test_loader:
                x, y, id = batch
                x = x.cuda()
                raw_labels = model(x).max(1)[1].cpu().numpy()
                for i, l in zip(id, raw_labels):
                    writer.writerow({'id': i, 'label': classes_dict[l]})

    elif submission_type == 'multi':

        root_models = './models/'
        model_list = ['0.94.pthe', 'l2_0.926.pthe', '0.926.pthe', '0.939.pthe',
                      '0.938.pthe', 'retrain_0.938.pthe']
        model_list = [root_models + m for m in model_list]

        models = [CIFARResNet18(num_classes=2, k=3).cuda() for _ in model_list]

        for m, w in zip(models, model_list):
            m.load_state_dict(torch.load(w))
            m.eval()

        with open('submission_multi.csv', mode='w') as csv_file:
            fieldnames = ['id', 'label']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for batch in test_loader:
                x, y, id = batch
                x = x.cuda()
                group_labels = [m(x).max(1)[1].cpu().numpy() for m in models]
                raw_labels = []
                for i in range(len(id)):
                    n_p = [0, 0]
                    for j in range(len(group_labels)):
                        n_p[group_labels[j][i]] += 1
                    raw_labels.append(np.argmax(n_p))
                for i, l in zip(id, raw_labels):
                    writer.writerow({'id': i, 'label': classes_dict[l]})
