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
from guided_diffusion.bratsloader import BRATSDataset
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

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    args.in_ch = 4
    ds = MyTrainData(args.data_dir, train=False, img_scale=1, transform=False)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
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

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    nn = 0
    dice1 = 0
    dice2 = 0
    iou1 = 0
    iou2 = 0
    # dice3 = 0
    # iou3 = 0
    # dice4 = 0
    # dice5 = 0
    while len(all_images) * args.batch_size < args.num_samples:
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel

        slice_ID = path[0].split("/")[-1].split('.')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        samplelist = []
        # caloutlist = []
        # callist = []
        # finallist = []
        # finallist1 = []
        # mea = torch.zeros((1, 1, 256, 256))
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            # sample, x_noisy, org, cal, cal_out, pred_start = sample_fn(
            sample, x_noisy, org, pred_start = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                # step = args.diffusion_steps,
                # step=1000,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
            # co = th.tensor(cal_out)
            #
            # cal_out = th.tensor(cal_out)
            # cal_out = th.where(cal_out > 0.5, 1, 0)
            #
            # cal= th.tensor(cal)
            # cal = th.where(cal > 0.5, 1, 0)

            sample = th.clamp(sample, 0, 1)
            sample = th.where(sample > 0.5, 1, 0).squeeze()
            # print("sample",sample.device)
            # mea = mea + sample.cpu()
            # sample = th.tensor(sample).squeeze()

            pred_start = th.where(pred_start > 0.5, 1, 0).squeeze()
            samplelist.append(sample)
            # finallist.append(final)
            # caloutlist.append(cal_out)
            # callist.append(cal)

            # if args.debug:
            #     s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
            #     o = th.tensor(org)[:,:-1,:,:]
            #     c = th.tensor(cal).repeat(1, 3, 1, 1)
            #     co = co.repeat(1, 3, 1, 1)
            #     tup = (o,s,c,co)
            #     compose = th.cat(tup,0)
            #     vutils.save_image(compose, fp = args.out_dir +str(slice_ID)+'_output'+str(i)+".jpg", nrow = 1, padding = 10)
        nn = nn + 1
        # sample_return = torch.zeros((1, 1, 256, 256))
        # for i in range(2):
        #     # uncer_out += samplelist[i]
        #     # uncer_out = uncer_out / 5
        #     uncer = compute_uncer(samplelist[i]).cpu()
        #
        #     w = torch.exp(1 - uncer)
        #     print("w",w)
        #     # fin = w * samplelist[i].cpu()
        #     # fin = th.clamp(fin, 0, 1)
        #     # fin = th.where(fin > 0.5, 1, 0).squeeze()
        #     # finallist.append(fin.squeeze())
        #     sample_return += w * samplelist[i].cpu()
        # sample_return = torch.sigmoid(sample_return)
        # sample_return = th.where(sample_return > 0.5, 1, 0).squeeze()
        # sample_return = torch.zeros((1, 1, 256, 256))
        #
        # for index in range(1000):
        #     #
        #     uncer_out = 0
        #     for i in range(5):
        #         uncer_out += finallist[i]["all_model_outputs"][index]
        #     uncer_out = uncer_out / 5
        #     uncer = compute_uncer(uncer_out).cpu()
        #
        #     w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 1000)) * (1 - uncer))
        #
        #     for i in range(5):
        #         sample_return += w * finallist[i]["all_samples"][index].cpu()
        # sample_return = torch.sigmoid(sample_return)
        # sample_return = th.where(sample_return > 0.5, 1, 0).squeeze()
        #
        # sample_return1 = torch.zeros((1, 1, 256, 256))
        # for index in range(100):
        #     #
        #     uncer_out1 = 0
        #     for i in range(5):
        #         uncer_out1 += finallist[i]["all_model_outputs"][index]
        #     uncer_out1 = uncer_out1 / 5
        #     uncer1 = compute_uncer(uncer_out1).cpu()
        #
        #     w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 100)) * (1 - uncer1))
        #
        #     for i in range(5):
        #         sample_return1 += w * finallist[i]["all_samples"][index].cpu()
        # sample_return1 = torch.sigmoid(sample_return1)
        # sample_return1 = th.where(sample_return1 > 0.5, 1, 0).squeeze()
        #
        # sample_return2 = torch.zeros((1, 1, 256, 256))
        # for index in range(900,1000):
        #     #
        #     uncer_out2 = 0
        #     for i in range(5):
        #         uncer_out2 += finallist[i]["all_model_outputs"][index]
        #     uncer_out2 = uncer_out2 / 5
        #     uncer2 = compute_uncer(uncer_out2).cpu()
        #
        #     w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 100)) * (1 - uncer2))
        #
        #     for i in range(5):
        #         sample_return2 += w * finallist[i]["all_samples"][index].cpu()
        # sample_return2 = torch.sigmoid(sample_return2)
        # sample_return2 = th.where(sample_return2 > 0.5, 1, 0).squeeze()
        # mea = mea / 5
        # mea = th.clamp(mea, 0, 1)
        # mea = th.where(mea>0.5, 1, 0).squeeze()
        ensres = th.stack(samplelist,dim=0)
        # print("sample:",sample.shape)#2,256,256
        ensres = th.mode(ensres,0)[0]#256,256
        # print("ensres:",ensres.shape)
        # calens = staple(th.stack(samplelist,dim=0)).squeeze()
        # caloutens = staple(th.stack(caloutlist,dim=0)).squeeze()
        # ensres = th.clamp(ensres, 0, 1)
        # ensres = th.where(ensres > 0.5, 1, 0)
        # calens = th.where(calens > 0, 1, 0)
        # caloutens = th.where(caloutens > 0, 1, 0)
        # sample = th.tensor(sample).squeeze()
        # final = th.stack(finallist, dim=0)
        # final = th.mode(final,0)[0]
        ddice2 = dice_score(pred_start.cpu().numpy(), m.squeeze().cpu().numpy())
        iiou2 = iou_score2(pred_start.cpu().numpy(), m.squeeze().cpu().numpy())
        logger.log(f"pred_xstart_dice: {ddice2}")
        dice2 = dice2 + ddice2
        iou2 = iou2 + iiou2
        logger.log(f"pred_xstart_dice:{dice2 / nn}")
        logger.log(f"pred_xstart_iou:{iou2 / nn}")
        pred_start = pred_start.cpu().numpy()
        out2 = Image.fromarray((pred_start * 255).astype(np.uint8))
        out2.save(args.out_dir+str(slice_ID)+"pred_xstart_"+".jpg")

        ddice1 = dice_score(ensres.cpu().numpy(), m.squeeze().cpu().numpy())
        iiou1 = iou_score2(ensres.cpu().numpy(), m.squeeze().cpu().numpy())
        logger.log(f"mode_dice: {ddice1}")
        dice1 = dice1 + ddice1
        iou1 = iou1 + iiou1
        logger.log(f"mode_final_dice:{dice1 / nn}")
        logger.log(f"mode_final_iou:{iou1 / nn}")
        ensres = ensres.cpu().numpy()
        out1 = Image.fromarray((ensres * 255).astype(np.uint8))  #
        out1.save(args.out_dir + str(slice_ID) + "mode_" + ".jpg")

        # ddice3 = dice_score(mea.cpu().numpy(), m.squeeze().cpu().numpy())
        # iiou3 = iou_score2(mea.cpu().numpy(), m.squeeze().cpu().numpy())
        # logger.log(f"mea_dice: {ddice3}")
        # dice3 = dice3 + ddice3
        # iou3 = iou3 + iiou3
        # logger.log(f"mea_final_dice:{dice3 / nn}")
        # logger.log(f"mode_final_iou:{iou3 / nn}")
        # mea = mea.cpu().numpy()
        # out3 = Image.fromarray((mea * 255).astype(np.uint8))
        # out3.save(args.out_dir + str(slice_ID) + "mean_" + ".jpg")
        #
        # ddice4 = dice_score(sample_return1.cpu().numpy(), m.squeeze().cpu().numpy())
        # logger.log(f"step_fusion100_dice: {ddice3}")
        # dice4 = dice4 + ddice4
        # logger.log(f"step_fusion_final_dice:{dice4 / nn}")
        # sample_return1 = sample_return1.cpu().numpy()
        # out4 = Image.fromarray((sample_return1 * 255).astype(np.uint8))
        # out4.save(args.out_dir + str(slice_ID) + "samplefirst100_" + ".jpg")
        #
        # ddice5 = dice_score(sample_return2.cpu().numpy(), m.squeeze().cpu().numpy())
        # logger.log(f"step_fusion900_dice: {ddice5}")
        # dice5 = dice5 + ddice5
        # logger.log(f"step_fusion_final_dice:{dice5 / nn}")
        # sample_return2 = sample_return2.cpu().numpy()
        # out5 = Image.fromarray((sample_return2 * 255).astype(np.uint8))
        # out5.save(args.out_dir + str(slice_ID) + "samplelast100_" + ".jpg")

        # ddice3 = dice_score(sample.cpu().numpy(), m.squeeze().cpu().numpy())
        # logger.log(f"sample_dice: {ddice3}")
        # dice3 = dice3 + ddice3
        # logger.log(f"sample_final_dice:{dice3 / nn}")
        # sample = sample.cpu().numpy()
        # out3 = Image.fromarray((sample * 255).astype(np.uint8))
        # out3.save(args.out_dir + str(slice_ID) + "sample_" + ".jpg")

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="/root/autodl-tmp/lxf/Mine/polyp/cell_division/test/",
        clip_denoised = False,
        predict_xstart = True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/root/autodl-tmp/lxf/Mine/scripts/outputs/unet++/0.9060.pt",
        num_ensemble=1,      #number of samples in the ensemble 5
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
