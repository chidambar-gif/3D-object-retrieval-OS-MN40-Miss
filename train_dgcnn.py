import os
import time
import json
from models.combine import pointcloud
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print("import successful")

from models import pointcloud
from loaders import OSMN40_train
from utils import split_trainval, AverageMeter, res2tab, acc_score, map_score
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


######### must config this #########
data_root = './data/OS-MN40-Miss'
####################################

# configure
n_class = 8
phase="DGCNN"
n_worker = 2
max_epoch =200
batch_size = 2
this_task = f"OS-MN40_{time.strftime('%Y-%m-%d-%H-%M-%S')}"


# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts'
save_dir.mkdir(parents=True, exist_ok=True)


def setup_seed():
    seed = time.time() % 1000_000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def train(data_loader, net, criterion, optimizer,epoch):
    print(f"Epoch {epoch}, Training...")
    Loss1=0
    freq=10
    net.train()
    loss_meter = AverageMeter()
    all_lbls, all_preds = [], []
    
    st = time.time()
    for i, (pt, lbl) in enumerate(data_loader):
        pt = pt.cuda()
        lbl = lbl.cuda()
              
        optimizer.zero_grad()
        out = net(pt)
      
 
        loss = criterion(out, lbl)
        Loss1=Loss1+loss.item()
        loss.backward()
        optimizer.step()
  

        _, preds = torch.max(out, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        if i % freq==0:
           print(f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}  ,total_loss for {freq} batches {Loss1:.4f}")
           Loss1=0


    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")


def validation(data_loader, net, epoch):
    print(f"Epoch {epoch}, Validation...")
    net.eval()
    all_lbls, all_preds = [], []
    fts_features= []

    st = time.time()
    for  pt , lbl in data_loader:
    
        pt = pt.cuda()
        lbl = lbl.cuda()

        out_obj, ft = net(pt, global_ft=True)
        

        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        fts_features.append(ft.detach().cpu().numpy())

    fts_features = np.concatenate(fts_features,axis=0)

    dist_mat = scipy.spatial.distance.cdist(fts_features, fts_features, "cosine")
    map_s = map_score(dist_mat, all_lbls, all_lbls)
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma,
        "map": map_s
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return map_s, res


def save_checkpoint(val_state, res, net: nn.Module):
    state_dict = net.state_dict()
    ckpt = dict(
        val_state=val_state,
        res=res,
        net=state_dict,
    )
    torch.save(ckpt, str(save_dir / 'DGCNN.pth'))
    with open(str(save_dir / 'DGCNN.meta'), 'w') as fp:
        json.dump(res, fp)


def main():
    setup_seed()
    # init train_loader and val_loader
    print("Loader Initializing...\n")
    train_list, val_list = split_trainval(data_root)

    train_data = OSMN40_train('train', train_list,phase)
    val_data = OSMN40_train('val', val_list,phase)

    print(f'train samples: {len(train_data)}')
    print(f'val samples: {len(val_data)}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                             num_workers=n_worker)

    print("Create new model")
    net = pointcloud(n_class)
    net = net.cuda()
    net = nn.DataParallel(net)
    
    

    
    #initialise optimizer
    optimizer = optim.SGD(net.parameters(),0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, 
                                                        eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_res, best_state = None, 0
    for epoch in range(max_epoch):
        train(train_loader, net, criterion, optimizer, epoch)
        lr_scheduler.step()
        if epoch!=0  and epoch % 5==0:
            with torch.no_grad():
                val_state, res = validation(val_loader, net, epoch)
            if val_state > best_state:
                print("saving model...")
                best_res, best_state = res, val_state
                save_checkpoint(val_state, res, net.module)

    print("\nTrain Finished!")
    tab_head, tab_data = res2tab(best_res)
    print(tab_head)
    print(tab_data)
    print(f'checkpoint can be found in {save_dir}!')
    return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
