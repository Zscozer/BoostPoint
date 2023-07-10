import pickle

import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from PointWOLF import PointWOLF
import data_utils as d_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class ScanObjectNN(Dataset):
    def __init__(self,args, num_points, root, train=True, **kwargs):
        super().__init__()
        self.train = train
        self.root = root
        self.num_points = num_points
        self.PointWolf = PointWOLF(args)

        if self.train:
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            # print(self.points.shape)     #(2309, 2048, 3)
            self.labels = np.array(h5['label'])
            # print(type(self.labels))
            # print(self.labels.shape)     #(2309,)
            h5.close()
        elif not self.train:
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label'])
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()  # type==torch.tensor 其中元素为float
        # if self.train:
        #     _, current_points = self.PointWolf(current_points)

        label = self.labels[idx]  # type==numpy.ndarray 其中元素为int
        label = np.array(label)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]


class ScanObjectNN_hardest(Dataset):
    def __init__(self, args,num_points, root, train=True, **kwargs):
        super().__init__()
        self.train = train
        self.root = root
        self.num_points = num_points
        self.PointWolf = PointWOLF(args)
        if self.train:
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            # print(self.points.shape)  (11416, 2048, 3)
            self.labels = np.array(h5['label'])
            print(self.labels.shape)
            # h5.close()  (11416,)
        elif not self.train:
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label'])
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        if self.train:
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()  # type==torch.tensor 其中元素为float
        # if self.train:
        #      _, current_points = self.PointWolf(current_points)

        label = self.labels[idx]  # type==numpy.ndarray 其中元素为int
        label = np.array(label)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]


def load_scanobjectnn_data_color(partition):
    # download()
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = "/home/zyk/WORKSPACE/zhou_honggong/ScanObjectNN/object_only_with_color/"

    save_path = BASE_DIR +"ScanObj_OBJONLY_"+ partition + '_2048pts_fps.dat'
    print('Load processed data from %s...' % save_path)
    with open(save_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)
    return list_of_points, list_of_labels


class ScanObjectNN_color(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_scanobjectnn_data_color(partition)
        self.data = np.array(self.data).astype(np.float32)
        self.label = np.array(self.label)
        self.num_points = num_points
        self.partition = partition

        # self.PointWOLF = PointWOLF(args)
    def __getitem__(self, item):
        pt_idxs = np.arange(0, self.data.shape[1])  # 2048
        if self.partition == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.data[item, pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()  # type==torch.tensor 其中元素为float

        label = self.label[item]  # type==numpy.ndarray 其中元素为int
        label = np.array(label)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return current_points, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    # 测试代码
    from torch.utils.data import DataLoader
    from torch.autograd import Variable

    train_dataset=ScanObjectNN(1024,'/home/zyk/WORKSPACE/zhou_honggong/ScanObjectNN/main_split')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    for i, data in enumerate(train_dataloader, 0):
        points, target = data
        points, target = points.cuda(), target.cuda()
        points, target = Variable(points), Variable(target)
        print(target)