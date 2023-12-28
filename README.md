# UDTransNet


This repo is the official implementation of
'[Narrowing the semantic gaps in U-Net with
learnable skip connections: The case of medical
image segmentation](https://arxiv.org/abs/2312.15182)' which is an improved journal version of [UCTransNet](https://github.com/McGregorWwww/UCTransNet).

**The manuscript is now under review.**

![framework](https://github.com/McGregorWwww/UDTransNet/blob/main/docs/Framework.jpg)


## Requirements

Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage


### 1. Data Preparation
The GlaS dataset is in the 'datasets' folder.
ISIC-2018 and Synapse datasets can be downloaded in following links:
* ISIC-2018 Dataset - [Link](https://challenge2018.isic-archive.com/task1/training/)
* Synapse Dataset - [Link](https://drive.google.com/file/d/1vxZ_eqqyycFva3luuDKZSTtyfd8-Uv3B/view?usp=sharing)

Then prepare the datasets in the following format for easy use of the code:
```angular2html
├── datasets
│   ├── GlaS
│   │   ├── Test_Folder
│   │   │   ├── img
│   │   │   └── labelcol
│   │   └── Train_Folder
│   │       ├── img
│   │       └── labelcol
│   ├── ISIC18
│   │   └── Train_Folder
│   │       ├── img
│   │       └── labelcol
│   └── Synapse
│       ├── lists
│       │   └── lists_Synapse
│       ├── test_vol_h5
│       └── train_npz
```

### 2. Training
We use five-fold cross validation strategy to train all the models on all the three datasets.

The first step is to change the settings in ```Config.py```,
all the configurations including learning rate, batch size and etc. are 
in it.

We optimize the convolution parameters 
in U-Net and the DAT parameters together with a single loss.
Run:
```angular2html
python train_kfold.py
```
The results including log files, model weights, etc., are in '[TaskName]_kfold' folder, e.g., 'GlaS_kfold'.


### 3. Testing
For GlaS and Synapse, we test the models of five folds and take the average score on the **test set**.

For ISIC'18, since the annotation of test set is not publicly available, we test the model of each fold on each **validation set**.
#### 3.1. Get Pre-trained Models
Here, we provide pre-trained weights of five folds on the three datasets, 
if you do not want to train the models by yourself, you can download them in this [Google Drive link](https://drive.google.com/drive/folders/1o1fRb10uptjGDAowTInH_7L4BmBGtCsf?usp=sharing).

#### 3.2. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase.

Then, for GlaS and Synapse, run:
```angular2html
python test_kfold.py
```
For ISIC, run:
```angular2html
python test_each_fold.py
```
You can get the Dice and IoU scores and the visualization results. 



## Codes Used in Our Experiments

* [UNet++](https://github.com/qubvel/segmentation_models.pytorch)
* [Attention U-Net](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
* [MultiResUNet](https://github.com/makifozkanoglu/MultiResUNet-PyTorch)
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)
* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [MCTrans](https://github.com/JiYuanFeng/MCTrans)
* [UCTransNet](https://github.com/McGregorWwww/UCTransNet)


<!--
## Citations

If this code is helpful for your study, please cite:
```
@misc{wang2021uctransnet,
      title={UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise Perspective with Transformer}, 
      author={Haonan Wang and Peng Cao and Jiaqi Wang and Osmar R. Zaiane},
      year={2021},
      eprint={2109.04335},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
-->

## Contact 
Haonan Wang ([haonan1wang@gmail.com](haonan1wang@gmail.com))
