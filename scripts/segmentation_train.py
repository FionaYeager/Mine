import sys
import torch
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from torch.utils.data import DataLoader
from guided_diffusion.resample import create_named_schedule_sampler
# from guided_diffusion.bratsloader import BRATSDataset
# from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.dataloader import MyTrainData
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
# from visdom import Visdom
# viz = Visdom(port=8850)
import torchvision.transforms as transforms
# torch.cuda.set_device(0)

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist(args) #分布式训练
    logger.configure(dir = args.out_dir) #输出文件夹：output

    logger.log("creating data loader...")

    # 数据集
    # if args.data_name == 'ISIC':
    #     tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    #     transform_train = transforms.Compose(tran_list)
    #
    #     ds = ISICDataset(args, args.data_dir, transform_train)
    #     args.in_ch = 4
    # elif args.data_name == 'BRATS':
    #     tran_list = [transforms.Resize((args.image_size,args.image_size)),]
    #     transform_train = transforms.Compose(tran_list)
    #
    #     ds = BRATSDataset(args.data_dir, transform_train, test_flag=False)
    #     args.in_ch = 5
    # datal= th.utils.data.DataLoader(
    #     ds,
    #     batch_size=args.batch_size,
    #     shuffle=True)
    args.in_ch = 4
    train_data = MyTrainData(args.data_dir, train=True, img_scale=1, transform=True)
    val_data = MyTrainData(args.val_dir , train=False, img_scale=1, transform=False)
    # n_val = len(val_data)
    # print("n_val:",n_val)
    # n_train = len(train_data)
    # print("train-data:",train_data)
    datal = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valdatal = DataLoader(val_data, batch_size=1, shuffle=True)
    # val_dl = th.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    data = iter(datal)
    valdata = iter(datal)

    logger.log("creating model and diffusion...")

    # 返回模型和扩散模型
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    # print(model)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        # valdata = valdata,
        dataloader=datal,
        # valdataloader=valdatal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'basedata_new',
        data_dir="/root/autodl-tmp/lxf/Mine/polyp/brainMRI/train/",
        val_dir="/root/autodl-tmp/lxf/Mine/polyp/brainMRI/test/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=1e-8,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        maxdice = 0,
        multi_gpu = None, #"0,1,2"
        out_dir='outputs/unet++/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
