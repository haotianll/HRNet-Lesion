import _init_paths
import models
import datasets

import argparse
import logging
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from config import config, update_config
from core.binary_loss import BinaryLoss
from core.lesion_evaluate import evaluate
from utils import distributed as dist
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--checkpoint', help='checkpoint file', required=True, type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset, shuffle=True):
    if dist.is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    gpus = list(range(dist.get_world_size()))

    # build model
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        image_scale=config.TEST.IMAGE_SIZE,
        pad_size=config.TEST.PAD_SIZE,
        mean=config.DATASET.MEAN,
        std=config.DATASET.STD,
        base_size=config.TEST.BASE_SIZE,
        downsample_rate=1,
        split='test')

    test_sampler = get_sampler(test_dataset, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    # criterion
    criterion = BinaryLoss(loss_type='dice', smooth=1e-5)
    model = FullModel(model, criterion)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    if args.local_rank <= 0:
        logger.info(model)

    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location={'cuda:0': 'cpu'})
        model.module.model.load_state_dict(
            {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
        logger.info("=> loaded checkpoint: {}".format(args.checkpoint))
        if distributed:
            torch.distributed.barrier()
    else:
        logger.info("=> checkpoint is not exist: {}".format(os.path.abspath(args.checkpoint)))
        return

    valid_loss, results, result_str = evaluate(config, testloader, model, test_dataset)
    if args.local_rank <= 0:
        logging.info('Eval Loss: {:.3f}'.format(valid_loss))
        logging.info(result_str)


if __name__ == '__main__':
    main()
