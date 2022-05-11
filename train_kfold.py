import os
import torch
import torch.optim
from torch.backends import cudnn
from tensorboardX import SummaryWriter
import numpy as np
import random
from utils import BinaryDiceBCE,MultiClassDiceCE, CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from Load_Dataset import RandomGenerator,ValGenerator, ImageToImage2D_kfold
from torch.utils.data import DataLoader
from nets.UNet import UNet,R34_UNet
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config
import logging
import warnings
from train_one_epoch import train_one_epoch
warnings.filterwarnings("ignore")
import Config as config

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type
    if best_model:
        filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + 'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          Main Loop
#=================================================================================
##################################################################################
def main_loop(train_loader,val_loader, batch_size=config.batch_size, model_type='', fold=0, tensorboard=True, kfold=0):

    lr = config.learning_rate
    logger.info(model_type)
    if model_type == 'UNet':
        model = UNet(n_channels=config.n_channels,n_classes=config.n_labels)
    elif model_type == 'R34_UNet':
        model = R34_UNet(n_channels=config.n_channels,n_classes=config.n_labels)
    elif model_type == 'UDTransNet':
        config_vit = get_model_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer head dim: {}'.format(config_vit.transformer.embedding_channels))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = UDTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels,img_size=config.img_size)

    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()

    if config.n_labels == 1:
        criterion = BinaryDiceBCE(dice_weight=1,BCE_weight=1)
    else:
        criterion = MultiClassDiceCE(num_classes=config.n_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Choose optimize

    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)
    else:
        lr_scheduler =  None

    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= {} | Fold [{}/{}] | Epoch [{}/{}] ========='.format(config.model_name,fold,kfold, epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, fold, kfold, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                          optimizer, writer, epoch, lr_scheduler,fold,kfold,logger)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch+1 > 1:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path+"fold_"+str(fold)+"/")
            else:pass
        elif val_dice == 0:
            best_epoch = epoch + 1
            logger.info('\t Reset count number')
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count >  config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return max_dice


if __name__ == '__main__':
    deterministic = False # set `True' can make the results reproducible, but costs more training time
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    if config.task_name == "Synapse":
        filelists = os.listdir(config.train_dataset)
    else:
        filelists = os.listdir(config.train_dataset+"img")

    filelists = np.array(filelists)
    kfold = config.kfold
    kf = KFold(n_splits=kfold, shuffle=True, random_state=config.seed)
    dice_list = []
    iou_list = []

    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
        train_filelists = filelists[train_index]
        val_filelists = filelists[val_index]
        np.savetxt(config.save_path+"val_fold_"+str(fold+1)+".txt", val_filelists,'%s')
        logger.info("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

        train_tf= RandomGenerator(output_size=[config.img_size, config.img_size2])

        val_tf = ValGenerator(output_size=[config.img_size, config.img_size2])
        train_dataset = ImageToImage2D_kfold(config.train_dataset,
                                             train_tf,
                                             image_size=config.img_size,
                                             filelists=train_filelists,
                                             task_name=config.task_name)
        val_dataset = ImageToImage2D_kfold(config.train_dataset,
                                           val_tf,
                                           image_size=config.img_size,
                                           filelists=val_filelists,
                                           task_name=config.task_name)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                worker_init_fn=worker_init_fn,
                                num_workers=8,
                                pin_memory=True)

        dice = main_loop(train_loader,val_loader, model_type=config.model_name, fold=fold+1, tensorboard=True, kfold=kfold)
        dice_list.append(dice.item())

    dice=0.0
    for j in range(len(dice_list)):
        logging.info("fold {0}: {1:2.4f}".format(j+1, dice_list[j]))
        dice+=dice_list[j]
    logging.info("mean dice: {:.4f} \n".format(dice/kfold))







