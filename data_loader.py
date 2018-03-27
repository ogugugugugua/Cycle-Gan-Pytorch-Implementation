import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch
import copy
import os
import shutil
def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

def reorganize(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'link_trainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'link_trainB')
    dirs['testA'] = os.path.join(dataset_dir, 'link_testA')
    dirs['testB'] = os.path.join(dataset_dir, 'link_testB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], '0'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], '0'))

    return dirs


class ItemPool(object):
    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """`in_items` is a list of item."""
        if self.max_num <= 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


#data size
load_size = 286
crop_size = 256
batch_size = 1
dataset_dir = './datasets/horse2zebra'
#maybe './datasets/horse2zebra'

def get_loader():
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.Scale(load_size),
         transforms.RandomCrop(crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    dataset_dirs = reorganize(dataset_dir)
    a_data = datasets.ImageFolder(dataset_dirs['trainA'], transform=transform)
    b_data = datasets.ImageFolder(dataset_dirs['trainB'], transform=transform)
    a_test_data = datasets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = datasets.ImageFolder(dataset_dirs['testB'], transform=transform)
    a_loader = torch.utils.data.DataLoader(a_data, batch_size=batch_size, shuffle=True, num_workers=4)
    b_loader = torch.utils.data.DataLoader(b_data, batch_size=batch_size, shuffle=True, num_workers=4)
    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=3, shuffle=True, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=3, shuffle=True, num_workers=4)


    return a_loader,b_loader,a_test_loader,b_test_loader

def fake_pool():
    a_fake_pool = ItemPool()
    b_fake_pool = ItemPool()
    return a_fake_pool,b_fake_pool

def save_checkpoint(state, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))
