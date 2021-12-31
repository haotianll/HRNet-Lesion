import _init_paths
import models
import datasets

import argparse
import os
import pprint
import timeit
import time
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim

from config import config, update_config
from core.function import reduce_tensor
from tensorboardX import SummaryWriter
from utils.utils import create_logger, FullModel, AverageMeter
from utils import distributed as dist

from core.binary_loss import BinaryLoss
from core.lesion_evaluate import evaluate
from utils.lesion_utils import adjust_learning_rate


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--resume-from', type=str, default=None)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset, shuffle=True):
    if dist.is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        current_lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter + cur_iters, min_lr=config.TRAIN.LR_MIN)
        writer.add_scalar('lr', current_lr, i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {}, Loss: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # build model
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        rotate=config.TRAIN.ROTATE,
        base_size=config.TRAIN.BASE_SIZE,
        image_scale=config.TRAIN.IMAGE_SIZE,
        ratio_range=config.TRAIN.RATIO_RANGE,
        crop_size=config.TRAIN.CROP_SIZE,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
        scale_factor=config.TRAIN.SCALE_FACTOR,
        mean=config.DATASET.MEAN,
        std=config.DATASET.STD,
        split='train'
    )

    train_sampler = get_sampler(train_dataset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    extra_epoch_iters = 0
    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('datasets.' + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.EXTRA_TRAIN_SET,
            num_samples=None,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            rotate=config.TRAIN.ROTATE,
            base_size=config.TRAIN.BASE_SIZE,
            image_scale=config.TRAIN.IMAGE_SIZE,
            ratio_range=config.TRAIN.RATIO_RANGE,
            crop_size=config.TRAIN.CROP_SIZE,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
            scale_factor=config.TRAIN.SCALE_FACTOR,
            mean=config.DATASET.MEAN,
            std=config.DATASET.STD,
            split='train')
        extra_train_sampler = get_sampler(extra_train_dataset, shuffle=True)
        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=batch_size,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)
        extra_epoch_iters = int(extra_train_dataset.__len__() /
                                config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

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

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR},
                      {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
        )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    last_epoch = 0

    if args.resume_from:
        model_state_file = args.resume_from
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            last_epoch = checkpoint['epoch']

            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    for epoch in range(last_epoch, end_epoch):
        current_trainloader = extra_trainloader if epoch >= config.TRAIN.END_EPOCH else trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        if (epoch + 1) % config.TRAIN.EVAL_INTERVAL == 0:
            valid_loss, results, result_str = evaluate(config, testloader, model, test_dataset, writer_dict)
            if args.local_rank <= 0:
                logging.info('Eval Loss: {:.3f}'.format(valid_loss))
                logging.info(result_str)

        if epoch >= config.TRAIN.END_EPOCH:
            train(config, epoch - config.TRAIN.END_EPOCH,
                  config.TRAIN.EXTRA_EPOCH, extra_epoch_iters,
                  config.TRAIN.EXTRA_LR, extra_iters,
                  extra_trainloader, optimizer, model, writer_dict)
        else:
            train(config, epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        if args.local_rank <= 0 and (epoch + 1) % 75 == 0:
            logger.info(
                '=> saving checkpoint to {}'.format(os.path.join(final_output_dir, f'epoch_{epoch + 1}.pth.tar')))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, f'epoch_{epoch + 1}.pth.tar'))

    valid_loss, results, result_str = evaluate(config, testloader, model, test_dataset, writer_dict)
    if args.local_rank <= 0:
        logging.info('Eval Loss: {:.3f}'.format(valid_loss))
        logging.info(result_str)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % int((end - start) / 3600))
        logger.info('Done')


if __name__ == '__main__':
    main()
