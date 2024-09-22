import os
import torch
import wandb
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from data.dataset import *
from torch.utils.data import DataLoader
from train_test import trainBaselineModel, trainDeformPathomicModel
from utils.yaml_config_hook import yaml_config_hook
from utils.sync_batchnorm import convert_model
from models.model import define_net, define_scheduler, define_optimizer


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # training set
    train_dataset_IvYGAP = IvYGAP_Dataset(phase='Train', args=args)
    input_size_omic_IvYGAP = args.input_size_omic
    input_size_omic_tumor_IvYGAP = args.input_size_omic_tumor
    input_size_omic_immune_IvYGAP = args.input_size_omic_immune
    
    train_dataset_TCGA = TCGA_Dataset(phase='Train', args=args)
    input_size_omic_TCGA = args.input_size_omic
    input_size_omic_tumor_TCGA = args.input_size_omic_tumor
    input_size_omic_immune_TCGA = args.input_size_omic_immune
        
    input_size_omic = args.input_size_omic
    input_size_omic_tumor = args.input_size_omic_tumor
    input_size_omic_immune = args.input_size_omic_immune
    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_IvYGAP, train_dataset_TCGA])

    # set sampler for parallel training
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    if rank == 0:
        if args.novalset:
            test_dataset_IvYGAP = IvYGAP_Dataset(phase='Test', args=args)
            test_dataset_TCGA = TCGA_Dataset(phase='Test', args=args)
            test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        else:
            test_dataset_IvYGAP = IvYGAP_Dataset(phase='Test', args=args)
            test_dataset_TCGA = TCGA_Dataset(phase='Test', args=args)
            test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            val_dataset_IvYGAP = IvYGAP_Dataset(phase='Val', args=args)
            val_dataset_TCGA = TCGA_Dataset(phase='Val', args=args)
            val_dataset = torch.utils.data.ConcatDataset([val_dataset_IvYGAP, val_dataset_TCGA])
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = None
        val_loader = None

    if args.novalset:
        loaders = (train_loader, test_loader)
    else:
        loaders = (train_loader, val_loader, test_loader)

    # # model init
    model = define_net(args).cuda()
    
    # reload model
    if args.reload:
        # model_fp = os.path.join(
        #     args.checkpoints, "epoch_{}_.pth".format(args.epochs)
        # )
        # model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        model_fp = os.path.join(
            args.checkpoints, "best_modal.pth"
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        

    model = model.to(args.device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = define_optimizer(args, model)
    scheduler = define_scheduler(args, optimizer)

    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)

    else:
        if args.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
            # model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
            # model._set_static_graph()

    if args.mode == 'deformpathomic':
        print("training deformpathomic model")
        trainDeformPathomicModel(model, loaders, optimizer, scheduler, wandb_logger, args)
    else:
        print("training Baseline model")
        trainBaselineModel(model, loaders, optimizer, scheduler, wandb_logger, args)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/config_mine.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visiable_device
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb if not in debug mode
    if not args.debug:
        wandb.login(key="#")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="MMD_on_%s"%args.dataset,
            notes="MMD 2023",
            tags=["TMI23", "Multi-modal Representation Learning"],
            config=config
        )
    else:
        wandb_logger = None


    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)
