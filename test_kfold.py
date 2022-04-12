from medpy import metric
from scipy.ndimage import zoom
import torchvision.transforms
import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D_kfold
from torch.utils.data import DataLoader
import warnings
import time
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.UNet import *
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config

import os
from utils import *
import cv2

def vis_save_synapse(input_img, pred, mask, save_path):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    if len(np.unique(mask)) > 2:
    # if True:
        # input_img=input_img.astype(np.uint8)
        # input_img = input_img * 255
        # input_img=input_img.astype(np.uint8)
        # input_img = cv2.cvtColor(input_img,cv2.COLOR_GRAY2BGR)
        input_img = input_img.convert('RGB')
        if pred is not None:
            pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
            input_img = np.where(pred==1, np.full_like(input_img, blue  ), input_img)
            input_img = np.where(pred==2, np.full_like(input_img, green ), input_img)
            input_img = np.where(pred==3, np.full_like(input_img, red   ), input_img)
            input_img = np.where(pred==4, np.full_like(input_img, cyan  ), input_img)
            input_img = np.where(pred==5, np.full_like(input_img, pink  ), input_img)
            input_img = np.where(pred==6, np.full_like(input_img, yellow), input_img)
            input_img = np.where(pred==7, np.full_like(input_img, purple), input_img)
            input_img = np.where(pred==8, np.full_like(input_img, orange), input_img)
        else:
            # mask = mask.convert('RGB')
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            input_img = np.where(mask==1, np.full_like(input_img, blue  ), input_img)
            input_img = np.where(mask==2, np.full_like(input_img, green ), input_img)
            input_img = np.where(mask==3, np.full_like(input_img, red   ), input_img)
            input_img = np.where(mask==4, np.full_like(input_img, cyan  ), input_img)
            input_img = np.where(mask==5, np.full_like(input_img, pink  ), input_img)
            input_img = np.where(mask==6, np.full_like(input_img, yellow), input_img)
            input_img = np.where(mask==7, np.full_like(input_img, purple), input_img)
            input_img = np.where(mask==8, np.full_like(input_img, orange), input_img)

        input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, input_img)


def show_ens(predict_save,input_img, labs, save_path):
    fig, ax = plt.subplots()
    plt.imshow(predict_save, cmap='gray')
    plt.axis("off")
    height, width = predict_save.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        iou = metric.binary.jc(pred, gt)
        return dice, iou
    elif pred.sum()==0 and gt.sum()==0:
        return 1, 1
    else:
        return 0, 0

def show_image_with_dice(predict_save, labs, save_path):

    if config.n_labels == 1:
        tmp_lbl = (labs).astype(np.float32)
        tmp_3dunet = (predict_save).astype(np.float32)
        dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)

        iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
        return dice_pred, iou_pred
    else:
        tmp_lbl = (labs).astype(np.float32)
        tmp_3dunet = (predict_save).astype(np.float32)
        metric_list = []
        for i in range(1, config.n_labels):
            metric_list.append(calculate_metric_percase(tmp_3dunet == i, tmp_lbl == i))
        metric_list = np.array(metric_list)

        dice_pred = np.mean(metric_list, axis=0)[0]
        iou_pred = np.mean(metric_list, axis=0)[1]
        dice_class = metric_list[:,0]

        return dice_pred, iou_pred, dice_class

def vis_and_save_heatmap(ensemble_models, input_img, img_RGB, labs,lab_img, vis_save_path):
    outputs = []
    dice_pred, iou_pred = [],[]
    for model_ in ensemble_models:
        output = model_(input_img.cuda())
        pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
        predict_save = pred_class[0].cpu().data.numpy()
        outputs.append(predict_save)
        dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
        dice_pred.append(dice_pred_tmp)
        iou_pred.append(iou_tmp)

    predict_save = np.array(outputs).mean(0)
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    predict_save = np.where(predict_save>0.5,1,0)
    show_ens(predict_save, img_RGB, lab_img, save_path=vis_save_path+'_pred5f_'+model_type+'.jpg')
    return dice_pred, iou_pred

def test_Synapse(ensemble_models, input_img, labs, vis_save_path):
    input_img = input_img.permute((1,0,2,3))
    labs = labs.permute((1,0,2,3))
    dice_pred_all = np.zeros(maxi)
    iou_pred_all = np.zeros(maxi)
    dice_class_all = np.zeros((maxi,8))
    num = input_img.size(0)
    for idx in range(num):
        res_vis = []
        dice_pred, iou_pred, dice_class = [],[],[]

        input_512, lab_512 = torchvision.transforms.functional.to_pil_image(input_img[idx]), torchvision.transforms.functional.to_pil_image(labs[idx])
        x, y = input_512.size
        input_vis = zoom(input_512, (224 / x, 224 / y), order=3)  # why not 3?
        label = zoom(lab_512, (224 / x, 224 / y), order=0)
        input = torchvision.transforms.functional.to_tensor(input_vis).unsqueeze(0)
        lab = np.array(label, np.uint8)
        lab_512 = np.array(lab_512, np.uint8)

        for model_ in ensemble_models:
            output = model_(input.cuda())
            predict_save = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            predict_save = predict_save.cpu().data.numpy()
            res_vis.append(output)
            dice_pred_tmp, iou_tmp, dice_class_tmp = show_image_with_dice(predict_save, lab, save_path=None)
            dice_pred.append(dice_pred_tmp)
            iou_pred.append(iou_tmp)
            dice_class.append(dice_class_tmp)
        res_vis = torch.cat(res_vis,dim=0)
        # print(res_vis.size())
        predict_save = torch.argmax(torch.softmax(res_vis.mean(0), dim=0), dim=0).cpu().data.numpy().astype(np.uint8)
        # print(predict_save.shape)
        predict_save_512 = zoom(predict_save, (512 / 224, 512 / 224), order=0)
        vis_save_synapse(input_512, predict_save_512, lab_512, save_path=vis_save_path+'_'+str(idx)+'_'+model_type+'.jpg')
        vis_save_synapse(input_512, None, lab_512, save_path=vis_save_path+'_'+str(idx)+'_gt.jpg')
        dice_pred_all += np.array(dice_pred)
        iou_pred_all += np.array(iou_pred)
        dice_class_all += np.array(dice_class)
    dice_pred_all /= num
    iou_pred_all /= num
    dice_class_all /= num

    return dice_pred_all, iou_pred_all, dice_class_all



if __name__ == '__main__':
    ## PARAMS
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ensemble_models=[]
    test_session = config.test_session

    for i in range(0,5):
        if config.task_name is "GlaS":
            test_num = 80
            model_type = config.model_name
            model_path = "./GlaS_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/best_model-"+model_type+".pth.tar"

        elif config.task_name is "Synapse":
            test_num = 12
            model_type = config.model_name
            model_path = "./Synapse_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(i+1)+"/best_model-"+model_type+".pth.tar"

        save_path    = config.task_name +'/'+ model_type +'/' + test_session + '/'

        att_vis_path = "./" + config.task_name + '_visualize_test/'

        if not os.path.exists(att_vis_path):
            os.makedirs(att_vis_path)

        maxi = 5
        if not os.path.exists(model_path):
            maxi = i
            print("====",maxi, "models loaded ====")
            break
        checkpoint = torch.load(model_path, map_location='cuda')

        if model_type == 'UNet':
            model = UNet(n_channels=config.n_channels,n_classes=config.n_labels)
        elif model_type == 'R34_UNet':
            model = R34_UNet(n_channels=config.n_channels,n_classes=config.n_labels)
        elif model_type == 'UDTransNet':
            config_vit = get_model_config()
            model = UDTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels, img_size=config.img_size)

        else: raise TypeError('Please enter a valid name for the model type')

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0,1,2,3])

        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded !')
        model.eval()
        ensemble_models.append(model)

    if config.n_labels == 1:
        filelists = os.listdir(config.test_dataset+"/img")
    else:
        filelists = os.listdir(config.test_dataset)
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D_kfold(config.test_dataset,
                                        tf_test,
                                        image_size=config.img_size,
                                        task_name=config.task_name,
                                        filelists=filelists,
                                        split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    dice_pred = np.zeros((maxi))
    iou_pred = np.zeros((maxi))
    dice_class = np.zeros((maxi,8))
    dice_ens = 0.0
    dice_5folds = []
    iou_5folds = []
    end = time.time()
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']

            if config.n_labels ==1:
                arr=test_data.numpy()
                arr = arr.astype(np.float32())
                lab=test_label.data.numpy()
                img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255

                fig, ax = plt.subplots()
                plt.imshow(img_lab, cmap='gray')
                plt.axis("off")
                height, width = config.img_size, config.img_size
                fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(att_vis_path+str(i)+"_lab.jpg", dpi=300)
                plt.close()

                img_RGB = cv2.imread(config.test_dataset+"img/"+names[0],1)
                img_RGB = cv2.resize(img_RGB,(config.img_size,config.img_size))


                lab_img = cv2.imread(config.test_dataset+"labelcol/"+names[0][:-4]+".png",0)
                lab_img = cv2.resize(lab_img,(config.img_size,config.img_size))
                input_img = torch.from_numpy(arr)

                dice_pred_t,iou_pred_t = vis_and_save_heatmap(ensemble_models, input_img, img_RGB, lab, lab_img,
                                                              att_vis_path+str(i))

            else:
                dice_pred_t,iou_pred_t,dice_class_t = test_Synapse(ensemble_models, test_data, test_label, att_vis_path+str(i))

            dice_pred_t = np.array(dice_pred_t)
            iou_pred_t = np.array(iou_pred_t)

            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            if config.n_labels > 1:
                dice_class_t = np.array(dice_class_t)
                dice_class+=dice_class_t

            torch.cuda.empty_cache()
            pbar.update()
    inference_time = (time.time() - end)/test_num
    print("inference_time",inference_time)
    dice_pred = dice_pred/test_num * 100.0
    iou_pred = iou_pred/test_num * 100.0
    if config.n_labels > 1:
        dice_class = dice_class/test_num * 100.0
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    print ("dice_5folds:",dice_pred)
    print ("iou_5folds:",iou_pred)
    dice_pred_mean = dice_pred.mean()
    iou_pred_mean = iou_pred.mean()
    if config.n_labels > 1:
        dice_class_mean = dice_class.mean(0)
    dice_pred_std = np.std(dice_pred,ddof=1)
    iou_pred_std = np.std(iou_pred,ddof=1)
    print ("dice: {:.2f}+{:.2f}".format(dice_pred_mean, dice_pred_std))
    print ("iou: {:.2f}+{:.2f}".format(iou_pred_mean, iou_pred_std))
    if config.n_labels > 1:
        np.set_printoptions(formatter={'float': '{:.2f}'.format})
        print ("dice class:",dice_class_mean)




