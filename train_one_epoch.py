# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:14 下午
# @Author  : Haonan Wang
# @File    : Train_one_epoch.py
# @Software: PyCharm
import torch.optim
import os
import time
from utils import save_on_batch
import Config as config
import warnings
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, fold,kfold,
                  average_loss, average_time,
                  dice, average_dice, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Fold: [{0}/{1}] Epoch: [{2}][{3}/{4}]  '.format(fold, kfold, epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, fold, kfold, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()

        # ====================================================
        #             Compute loss
        # ====================================================
        preds = model(images)
        if config.n_labels>1:
            out_loss = criterion(preds, masks.long(), softmax=True)
        else:
            out_loss = criterion(preds, masks.float())

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()


        if config.n_labels>1:
            train_dice = criterion._show_dice(preds, masks.long(), softmax=True)
        else:
            train_dice = criterion._show_dice(preds, masks.float())

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val' and epoch is not 0:
            vis_path = config.visualize_path+ "fold_"+str(fold)+"/"+ str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader),
                          out_loss, fold,kfold,
                          average_loss, average_time,
                          train_dice, train_dice_avg, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg

