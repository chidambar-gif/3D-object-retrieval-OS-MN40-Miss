# 3D-object-retrieval-OS-MN40-Miss

# Setup
## Install Related Packages
This code is developed in Python 3.8.12 and pytorch1.8.1+cu102. You can install the required packages as follows.
``` bash 
conda create -n shrec2022 python=3.7
conda activate shrec2022
pip install -r requirements.txt
```

## Configure Path
By default, the datasets are placed under the "data" folder in the root directory. This code will create a new folder (name depends on the current time) to restore the checkpoint files under "cache/ckpts" folder for each run.
``` bash
├── cache
│   └── ckpts
│        ├── cdist.txt
│        ├── ckpt.meta
│        └── DGCNN.pth
│        └──MVCNN.pth
└── data
    ├── OS-MN40/
    └── OS-MN40-Miss/
```
# Modality-generation
With help of pymesh-lab and open3d libraries we generated multi-view images,pointcloud,mesh and voxels for [OS-MN40-Miss] data.

# To generate modalities
Move modality-generation.ipynb and binvox file into the data folder and run the modality-generation notebook.

## Models and implementation
We implement the our methodology  combining multi-modal backbone, as follows:
- image: [resnet18] with 24-view input
- point cloud: [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)

# Train and Validation

After modality-generation Run train_dgcnn.py and train_mvcnn.py seperately. The checkpoints are created for the both runs will be in the cache ckpts folder. 80% data in the train folder is used for traning and the rest is used for validation
``` bash
python train_dgcnn.py
python train_mvcnn.py
```

## Generate Distance Matrix
Run the get_mat.py by loading the checkpoints of above mentioned tasks:
``` bash
python get_mat.py
```
The generated cdist.txt can be found in the same folder of the specified checkpoints. 

Pretrained check points on OS-MN40-Miss data can be found [here](https://drive.google.com/drive/folders/1RbmZUXuq0XBowcHGYQM3R_OFkAlYVVFy?usp=sharing).
Move DGCNN.pth and MVCNN.pth to cache/ckpts directory.



