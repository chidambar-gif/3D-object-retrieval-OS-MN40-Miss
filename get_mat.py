import os
import time
import json
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from models import combine
from loaders import OSMN40_retrive
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

######### must config this #########
data_root = Path('../data/OS-MN40')
ckpt_path = 'cache/ckpts'
####################################

# configure the variables
dist_mat_path = Path(ckpt_path).parent / "cdist.txt"
phase="RETRIVE"
dist_metric='cosine'
batch_size = 6
n_worker = 8
n_class = 8


def extract(query_loader, target_loader, net):
    net.eval()
    print("Extracting....")

    q_fts_all = []
    t_fts_all = []
    
    print("loading query_objects")
    st = time.time()
    for img, pt ,flag in query_loader:
        img = img.cuda()
        pt = pt.cuda()
        flag =flag.squeeze(1).cuda()
        data = (img, pt,flag)
        ft = net(data, global_ft=True)
 

        q_fts_all.append(ft.detach().cpu().numpy())
       
    q_fts_uni = np.concatenate(q_fts_all, axis=0)
   
    print("loading target objects")
    for img, pt,flag in target_loader:
        img = img.cuda()
        pt = pt.cuda()
        flag =flag.squeeze(1).cuda()
        data = (img, pt,flag)
    
        ft = net(data, global_ft=True)
        t_fts_all.append(ft.detach().cpu().numpy())
        
    t_fts_uni = np.concatenate(t_fts_all, axis=0)
  
   

    print(f"Time Cost: {time.time()-st:.4f}")

    dist_mat = scipy.spatial.distance.cdist(q_fts_uni, t_fts_uni, dist_metric)
    np.savetxt(str(dist_mat_path), dist_mat)

def read_object_list(filename, pre_path):
    object_list = []
    with open(filename, 'r') as fp:
        for name in fp.readlines():
            if name.strip():
                object_list.append(str(pre_path/name.strip()))
    return object_list


def main():
    # init train_loader and test loader
    print("Loader Initializing...\n")
    query_list = read_object_list("query.txt", data_root / "query")
    target_list = read_object_list("target.txt", data_root / "target")
    query_data = OSMN40_retrive(query_list,phase)
    target_data = OSMN40_retrive(target_list,phase)
    print(f'query samples: {len(query_data)}')
    print(f'target samples: {len(target_data)}')
    query_loader = DataLoader(query_data, batch_size=batch_size, shuffle=False,
                                               num_workers=n_worker)
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=False,
                                             num_workers=n_worker)
    print(f"Loading model from {ckpt_path}")
    net = combine(n_class,ckpt_path)
    net = net.cuda()
    net = nn.DataParallel(net)

    # extracting
    with torch.no_grad():
        extract(query_loader, target_loader, net)

    print(f"cdis matrix can be find in path: {dist_mat_path.absolute()}")


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
