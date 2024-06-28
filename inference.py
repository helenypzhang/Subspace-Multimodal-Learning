import os
import torch
import wandb
import argparse
import numpy as np
from data.dataset import *
from torch.utils.data import DataLoader
from train_test import testDeformPathomicModel, testBaselineModel
from utils.yaml_config_hook import yaml_config_hook
from utils.sync_batchnorm import convert_model
from models.model import define_net


def main(gpu, args, wandb_logger):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_dataset_IvYGAP = IvYGAP_Dataset(phase='Test', args=args)
    test_dataset_TCGA = TCGA_Dataset(phase='Test', args=args)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    loaders = (None, test_loader)  # trainloader is not needed for testing

    # for individually testing two datasets:
    # val_dataset_IvYGAP = IvYGAP_Dataset(phase='Val', args=args)
    # val_dataset_TCGA = TCGA_Dataset(phase='Val', args=args)
    # valtest_dataset_IvYGAP = torch.utils.data.ConcatDataset([val_dataset_IvYGAP, test_dataset_IvYGAP])
    # valtest_dataset_TCGA = torch.utils.data.ConcatDataset([val_dataset_TCGA, test_dataset_TCGA])
    # valtest_dataset_two = torch.utils.data.ConcatDataset([valtest_dataset_IvYGAP, valtest_dataset_TCGA])
    # loader_IvYGAP = DataLoader(valtest_dataset_IvYGAP, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # loader_TCGA = DataLoader(valtest_dataset_TCGA, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # loader_two = DataLoader(valtest_dataset_two, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # loaders = (None, loader_TCGA)
    
    # Initialize model
    model = define_net(args).cuda()

    # model_fp = os.path.join(args.checkpoints, "#")s
    model_fp = "#"
    if os.path.isfile(model_fp):
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    else:
        raise FileNotFoundError("testing model not found at {}".format(model_fp))
        
    model = model.to(args.device)

    if args.mode == 'deformpathomic':
        testDeformPathomicModel(model, loaders, wandb_logger, args)
    else:
        testBaselineModel(model, loaders, wandb_logger, args)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/config_mine.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

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


    main(0, args, wandb_logger)
