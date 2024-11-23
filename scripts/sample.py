import argparse
import sys
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import torch

sys.path.append("..")
sys.path.append(".")
import sys
import random
sys.path.append(".")
from guided_diffusion.utils import dice_score,iou_score2
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.diff_unet import UNet
# from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import dice_coeff2
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
from guided_diffusion.dataloader import MyTrainData
# from guided_diffusion.bratsloader import MyTrainData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.spatial.distance import directed_hausdorff
import os
import os.path
from glob import glob
from sklearn.model_selection import train_test_split
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# torch.cuda.set_device(0)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def compute_uncer(pred_out):
    # pred_out = torch.sigmoid(pred_out)
    # pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

def Hausdorf(output, target):

    temp_preds = output.cpu().detach().numpy().squeeze(0)
    temp_target = target.cpu().detach().numpy().squeeze(0)
    dist1 = directed_hausdorff(temp_preds, temp_target)[0]
    dist2 = directed_hausdorff(temp_target, temp_preds)[0]
    hauss = max(dist1, dist2)

    return hauss

def hunxiao(preds, target):
    n_pred = preds.ravel()
    n_target = target.astype('int64').ravel()

    tn, fp, fn, tp = confusion_matrix(n_pred, n_target, labels=[0,1]).ravel()
    smooh = 1e-10
    sensitivity = tp/ (tp + fn + smooh)
    specificity = tn / (tn + fp + smooh)
    acc = (tp + tn) / (tn + tp + fp + fn + smooh)
    precision = tp / (tp + fp + smooh)
    # f1 = (2 * precision * sensitivity) / (precision + sensitivity + smooh)

    return sensitivity,specificity,acc,precision

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    args.in_ch = 4

    # img_ids = glob(os.path.join(args.data_dir, 'imgs', '*' + '.png'))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # img_ids.sort()
    # len_id = len(img_ids)
    # test_size = int((10 / 100) * len_id)
    # train_img_ids, test_img_ids = train_test_split(img_ids, test_size=test_size, random_state=42)
    # train_img_ids, val_img_ids = train_test_split(train_img_ids, test_size=test_size, random_state=42)

    # ds = MyTrainData("/root/autodl-tmp/lxf/Mine/polyp/brain/test/",img_ids=test_img_ids, train=False, img_scale=1, transform=False)
    ds = MyTrainData("/root/autodl-tmp/lxf/Mine/polyp/brain/test/", train=False, img_scale=1, transform=False)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=16,
        shuffle=False)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict
    model1 = UNet(4, 1)
    model1.load_state_dict(new_state_dict)

    model1.to(dist_util.dev())
    if args.use_fp16:
        model1.convert_to_fp16()
    model1.eval()
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    nn = 0
    dice1 = 0
    dice2 = 0
    dice3 = 0
    hauss = 0
    hauss1 = 0
    hauss2 = 0
    hauss3 = 0
    recall1 = 0
    recall2 = 0
    recall3 = 0
    spe1 = 0
    spe2 = 0
    spe3 = 0
    accuracy1 = 0
    accuracy2 = 0
    accuracy3 = 0
    precision1 = 0
    precision2 = 0
    precision3 = 0
    iou1 = 0
    iou2 = 0
    iou3 = 0
    while len(all_images) * args.batch_size < args.num_samples:
        b, m, path = next(data)  #should return an image from the dataloader "data"
        # b, m, slice_ID = next(data)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel

        slice_ID = path[0].split("/")[-1].split('.')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        samplelist = []
        mean = torch.zeros((img.shape[0], 1, 256, 256)).cuda()
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            # sample, x_noisy, org, cal, cal_out, pred_start = sample_fn(
            sample, x_noisy, org, pred_start = sample_fn(
                model1,
                (img.shape[0], 3, args.image_size, args.image_size), img,
                # step = args.diffusion_steps,
                # step=1000,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            sample = th.clamp(sample, 0, 1)
            # pred_start = th.clamp(pred_start, 0, 1)
            sample = th.where(sample > 0.5, 1, 0)#.squeeze()
            mean = sample + mean
            pred_start = th.where(pred_start > 0.5, 1, 0)#.squeeze()
            samplelist.append(sample.squeeze(1))
            # print("pred_start:", pred_start.shape)
            # print("m:", m.squeeze().shape)
            if i==0:
                d1 = d1 + dice_score(pred_start.cpu(), m)#.squeeze().squeeze().squeeze().squeeze().squeeze()
            if i==1:
                d2 = d2 + dice_score(pred_start.cpu(), m)
            if i==2:
                d3 = d3 + dice_score(pred_start.cpu(), m)
            if i==3:
                d4 = d4 + dice_score(pred_start.cpu(), m)
            if i==4:
                d5 = d5 + dice_score(pred_start.cpu(), m)
            # print("i:", i)

        nn = nn + 1
        # d1 = d1 / (nn)
        print("mdice1:", d1 / (nn))

        print("mdice2:", d2 / (nn))

        print("mdice3:", d3 / (nn))

        print("mdice4:", d4 / (nn))

        print("mdice5:", d5 / (nn))

        # mean = mean / 5
        # row = mean
        mean = th.where(mean > 0, 1, 0)#.squeeze()
        ensres = th.stack(samplelist, dim=0)#0
        # print("ensres:", ensres.shape)
        ensres = th.mode(ensres, 0)[0]#256,256
        # final = mean + ensres
        # final = th.clamp(final, 0, 1)
        # final = th.where(final > 0.5, 1, 0)
        # print("ensres:", ensres.shape)  # 2,256,256
        i = 0
        o = 0
        u = 0
        hhaus1 = 0
        hhaus2 = 0
        hhaus3 = 0
        for output3, mask_gt3 in zip(pred_start, m):#.squeeze(1)
            i = i + 1
            mask_pred3 = (output3 > 0.5).float()  # mask_pred.size()(1,256,256),mask_gt.size()(1,256,256)
            hhaus3 = hhaus3 + Hausdorf(mask_pred3, mask_gt3)
        for output2, mask_gt2 in zip(mean, m):
            o = o + 1
            mask_pred2 = (output2 > 0.5).float()  # mask_pred.size()(1,256,256),mask_gt.size()(1,256,256)
            hhaus2 = hhaus2 + Hausdorf(mask_pred2, mask_gt2)
        # print(ensres.shape)
        for output1, mask_gt1 in zip(ensres.unsqueeze(1), m):#ensemble1的话就是0
            u = u + 1
            mask_pred1 = (output1 > 0.5).float()  # mask_pred.size()(1,256,256),mask_gt.size()(1,256,256)
            hhaus1 = hhaus1 + Hausdorf(mask_pred1, mask_gt1)
        # hhaus2 = Hausdorf(mean, m.squeeze())
        hhaus2 = hhaus2 / o
        # hhaus1 = Hausdorf(ensres, m.squeeze())#2
        hhaus1 = hhaus1 / u
        hhaus3 = hhaus3 / i
        # hhaus3 = Hausdorf(pred_start, m.squeeze())
        # acm_p = torch.sigmoid(mask_pred)
        mask_nd = np.array(m.squeeze().cpu())
        # mask_s = mask_nd.astype('float32') / 255
        acm_p1 = np.int64(mean.squeeze().cpu().detach().numpy() > 0.5)
        acm_p2 = np.int64(ensres.squeeze().cpu().detach().numpy() > 0.5)
        acm_p3 = np.int64(pred_start.squeeze().cpu().detach().numpy() > 0.5)
        rrecall1, sspe1, aaccuracy1, pprecision1 = hunxiao(acm_p1, mask_nd)
        rrecall2, sspe2, aaccuracy2, pprecision2 = hunxiao(acm_p2, mask_nd)
        rrecall3, sspe3, aaccuracy3, pprecision3 = hunxiao(acm_p3, mask_nd)

        ddice2 = dice_score(mean.squeeze().cpu().numpy(), m.squeeze().cpu().numpy())
        iiou2 = iou_score2(mean.squeeze().cpu().numpy(), m.squeeze().cpu().numpy())
        logger.log(f"**********************")
        logger.log(f"n:{nn}")
        logger.log(f"mean_dice: {ddice2}")
        dice2 = dice2 + ddice2
        iou2 = iou2 + iiou2
        hauss2 = hauss2 + hhaus2
        recall2 = recall2 + rrecall2
        spe2 = sspe2 + spe2
        accuracy2 = aaccuracy2 + accuracy2
        precision2 = pprecision2 + precision2
        logger.log(f"mean_dice:{dice2 / nn}")
        logger.log(f"mean_iou:{iou2 / nn}")
        logger.log(f"mean_hauss:{hauss2 / nn}")
        logger.log(f"mean_recall:{recall2 / nn}")
        logger.log(f"mean_spe:{spe2 / nn}")
        logger.log(f"mean_precision:{precision2 / nn}")
        logger.log(f"mean_accuracy:{accuracy2 / nn}")
        # mean = mean.cpu().numpy()
        # out2 = Image.fromarray((mean * 255).astype(np.uint8))
        # out2.save(args.out_dir+str(slice_ID)+"mean_"+".jpg")

        ddice1 = dice_score(ensres.squeeze().cpu().numpy(), m.squeeze().cpu().numpy())
        iiou1 = iou_score2(ensres.squeeze().cpu().numpy(), m.squeeze().cpu().numpy())
        logger.log(f"**********************")
        logger.log(f"n:{nn}")
        logger.log(f"mode_dice: {ddice1}")
        dice1 = dice1 + ddice1
        iou1 = iou1 + iiou1
        hauss1 = hauss1 + hhaus1
        recall1 = recall1 + rrecall1
        spe1 = sspe1 + spe1
        accuracy1 = aaccuracy1 + accuracy1
        precision1 = pprecision1 + precision1
        logger.log(f"mode_final_dice:{dice1 / nn}")
        logger.log(f"mode_final_iou:{iou1 / nn}")
        logger.log(f"mode_hauss:{hauss1 / nn}")
        logger.log(f"mode_recall:{recall1 / nn}")
        logger.log(f"mode_spe:{spe1 / nn}")
        logger.log(f"mode_precision:{precision1 / nn}")
        logger.log(f"mode_accuracy:{accuracy1 / nn}")
        # ensres = m.squeeze().cpu().numpy()
        # out1 = Image.fromarray((ensres * 255).astype(np.uint8))  #
        # out1.save(args.out_dir + str(slice_ID) + "mode_" + ".jpg")

        ddice3 = dice_score(pred_start.squeeze().cpu().numpy(), m.squeeze().cpu().numpy())
        iiou3 = iou_score2(pred_start.squeeze().cpu().numpy(), m.squeeze().cpu().numpy())
        logger.log(f"**********************")
        logger.log(f"n:{nn}")
        logger.log(f"final_dice: {ddice3}")
        dice3 = dice3 + ddice3
        iou3 = iou3 + iiou3
        hauss3 = hauss3 + hhaus3
        recall3 = recall3 + rrecall3
        spe3 = sspe3 + spe3
        accuracy3 = aaccuracy3 + accuracy3
        precision3 = pprecision3 + precision3
        logger.log(f"final_dice:{dice3 / nn}")
        logger.log(f"final_iou:{iou3 / nn}")
        logger.log(f"final_hauss:{hauss3 / nn}")
        logger.log(f"final_recall:{recall3 / nn}")
        logger.log(f"final_spe:{spe3 / nn}")
        logger.log(f"final_precision:{precision3 / nn}")
        logger.log(f"final_accuracy:{accuracy3 / nn}")
        # pred_start = pred_start.cpu().numpy()
        # out3 = Image.fromarray((pred_start * 255).astype(np.uint8))
        # out3.save(args.out_dir + str(slice_ID) + "start_" + ".jpg")
def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="/root/autodl-tmp/lxf/Mine/polyp/brain/test/",
        clip_denoised = True,
        predict_xstart = True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        model_path="/root/autodl-tmp/lxf/Mine/scripts/outputs/unet++/savedmodel032000.pt",
        num_ensemble=5,      #number of samples in the ensemble 5
        gpu_dev = "0",
        # diffusion_steps=10,
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
