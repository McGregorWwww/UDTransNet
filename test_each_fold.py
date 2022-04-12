import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D_kfold
from torch.utils.data import DataLoader
import warnings
from sklearn.model_selection import KFold
import time
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.UNet import UNet,R34_UNet
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config
import os
from utils import *
import cv2


def show_ens(predict_save,input_img, labs, save_path):
    size =  512
    predict_save=cv2.resize(predict_save, (size,size))
    labs=cv2.resize(labs, (size,size))
    input_img=cv2.resize(input_img, (size,size))
    lbl_contour, hierarchy = cv2.findContours(labs.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(input_img, contours=lbl_contour, contourIdx=-1, color=(0, 255, 255),thickness=2)  # 红色 0 层
    pred_contour, hierarchy = cv2.findContours(predict_save.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(input_img, contours=pred_contour, contourIdx=-1, color=(0, 255, 0),thickness=2)  # 红色 0 层
    cv2.imwrite(save_path, input_img)


def show_image_with_dice(predict_save, labs):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs,lab_img, vis_save_path):
    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size2, config.img_size))
    labs = np.reshape(labs, (config.img_size2, config.img_size))
    dice_pred, iou_pred = show_image_with_dice(predict_save, labs)

    show_ens(predict_save, img_RGB, lab_img, save_path=vis_save_path+'_'+model_type+'.jpg')
    return dice_pred, iou_pred



def test_each_fold(val_filelists, model, test_num):
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size2])
    test_dataset = ImageToImage2D_kfold(config.train_dataset,
                                       tf_test,
                                       image_size=config.img_size,
                                       filelists=val_filelists,
                                       task_name=config.task_name)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    dice_pred_mean_1fold = 0.0
    iou_pred_mean_1fold = 0.0
    start = time.time()
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255

            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size2, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(att_vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()

            img_RGB = cv2.imread(config.train_dataset+"img/"+names[0],1)
            img_RGB = cv2.resize(img_RGB,(config.img_size,config.img_size2))
            lab_img = cv2.imread(config.train_dataset+"labelcol/"+names[0][:-4]+"_segmentation.png",0)
            lab_img = cv2.resize(lab_img,(config.img_size,config.img_size2))

            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, img_RGB, lab, lab_img,
                                                          att_vis_path+str(i))

            dice_pred_mean_1fold += dice_pred_t
            iou_pred_mean_1fold += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    inference_time = (time.time() - start)/test_num
    return inference_time, dice_pred_mean_1fold/test_num, iou_pred_mean_1fold/test_num



if __name__ == '__main__':
    ## PARAMS
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ensemble_models=[]
    test_session = config.test_session
    dice_pred = []
    iou_pred = []

    filelists = os.listdir(config.train_dataset+"img")
    filelists = np.array(filelists)
    kfold = config.kfold
    kf = KFold(n_splits=kfold, shuffle=True, random_state=config.seed)

    dice_list = []
    iou_list = []
    inference_time = 0.0

    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):

        model_type = config.model_name
        model_path = "./ISIC18_kfold/"+model_type+"/"+test_session+"/models/fold_"+str(fold+1)+"/best_model-"+model_type+".pth.tar"

        print(model_type)
        save_path    = config.task_name +'/'+ model_type +'/' + test_session + '/'

        att_vis_path = "./" + config.task_name + '_visualize_test/fold_' + str(fold+1) + '/'

        if not os.path.exists(att_vis_path):
            os.makedirs(att_vis_path)

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
        model.eval()

        val_filelists = filelists[val_index]
        test_num = len(val_filelists)

        inference_time_t, dice_pred_mean_1fold, iou_pred_mean_1fold = test_each_fold(val_filelists, model, test_num)
        print ("dice:{:.4f}".format(dice_pred_mean_1fold))
        print ("iou:{:.4f}".format(iou_pred_mean_1fold))
        print ("time:{:.3f}".format(inference_time_t))
        dice_pred.append(dice_pred_mean_1fold)
        iou_pred.append(iou_pred_mean_1fold)
        inference_time+=inference_time_t
    dice_pred = np.array(dice_pred)
    iou_pred = np.array(iou_pred)
    print("inference_time",inference_time/5.0)
    dice_pred = dice_pred * 100.0
    iou_pred = iou_pred * 100.0
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    print ("dice_5folds:",dice_pred)
    print ("iou_5folds:",iou_pred)
    dice_pred_mean = dice_pred.mean()
    iou_pred_mean = iou_pred.mean()
    dice_pred_std = np.std(dice_pred,ddof=1)
    iou_pred_std = np.std(iou_pred,ddof=1)
    print ("dice: {:.2f}+{:.2f}".format(dice_pred_mean, dice_pred_std))
    print ("iou: {:.2f}+{:.2f}".format(iou_pred_mean, iou_pred_std))




