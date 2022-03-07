from glob import glob
import torch
import torch.nn as nn
from .image import MVCNN
from .dgcnn import DGCNN
import torch.nn.functional as F
import os


class pointcloud(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_pt = DGCNN(n_class)

    def forward(self, data, global_ft=False):
        out_pt, ft_pt = self.model_pt(data)

        if global_ft:
            return out_pt,ft_pt
        else:
            return out_pt

  
class image(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img = MVCNN(n_class, n_view=24)
        
    def forward(self, data, global_ft=False):
        out_img, ft_img = self.model_img(data)

        if global_ft:
            return out_img,ft_img
        else:
            return out_img

class combine(nn.Module):
    def __init__(self,n_class,path):
        super().__init__()
        
        #call pretrained dgcnn and mvcnn models
        self.model_pt = pointcloud(n_class)
        self.model_img=image(n_class)
        self.path=path
        
        #load the respective checkpoints for pretrained models
        ckpt1=torch.load(self.path+"/DGCNN.pth")
        ckpt2=torch.load(self.path+"/MVCNN.pth")
    
        self.model_pt.load_state_dict(ckpt1['net'])
        self.model_img.load_state_dict(ckpt2['net'])

        self.model_pt=self.model_pt.cuda()
        self.model_img=self.model_img.cuda()

        self.model_pt_=nn.DataParallel(self.model_pt)
        self.model_img_=nn.DataParallel(self.model_img)
       
        
        #calling the objects of above defined clssess
        self.model_pt=self.model_pt.model_pt
        self.model_img=self.model_img.model_img

    def forward(self, data, global_ft=False):
        img,pt,flag=data
        flag=flag.unsqueeze(1).repeat([1,1,1024])
        flag=flag.squeeze(0)
            

        _, ft_img = self.model_img(img)

        _, ft_pt = self.model_pt(pt)
     
        ft_pt=ft_pt*flag
        

        global_feat=torch.cat((0.7692*ft_img,0.2307*ft_pt),axis=1) #weighted fusion of the features
        
        return global_feat
        

        

