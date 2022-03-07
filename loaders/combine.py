import torch
from pathlib import Path
from torch.utils.data import Dataset
from .image import load_img
from loaders import load_pt
import numpy as np



class OSMN40_train(Dataset):
    def __init__(self, phase, object_list,model):
        super().__init__()
        assert phase in ('train', 'val')
        self.phase = phase
        self.object_list = object_list
        self.model=model

    def __getitem__(self, index):
        p = Path(self.object_list[index]['path'])
        lbl = self.object_list[index]['label']

        if self.model=="DGCNN":
            pt,_= load_pt(p/'pointcloud', self.phase=='train')
            return pt,lbl
        if self.model=="MVCNN":
            img = load_img(p/'image', self.phase=='train')
            return img,lbl
        if self.model=="RETRIVE":
            img = load_img(p/'image', self.phase=='train')
            pt,flag = load_pt(p/'pointcloud', self.phase=='train')
            return img, pt,torch.from_numpy(np.array([flag]).astype(np.int64)),lbl

    def __len__(self):
        return len(self.object_list)


class OSMN40_retrive(Dataset):
    def __init__(self, object_list,model):
        super().__init__()
        self.object_list = object_list
        self.model=model

    def __getitem__(self, index):
        p = Path(self.object_list[index])

        if self.model=="DGCNN":
            pt,_= load_pt(p/'pointcloud')
            return pt
        if self.model=="MVCNN":
            img = load_img(p/'image')
            return img
        if self.model=="RETRIVE":
            img = load_img(p/'image')
            pt,flag= load_pt(p/'pointcloud')
            return img, pt,torch.from_numpy(np.array([flag]).astype(np.int64))


    def __len__(self):
        return len(self.object_list)


if __name__ == '__main__':
    pass
