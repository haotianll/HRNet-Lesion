import os
import numpy as np
import torch
from sklearn.metrics import auc
from torch.nn import functional as F

from core.function import reduce_tensor
from utils import distributed as dist
from utils.utils import AverageMeter

np.seterr(invalid='ignore')


def reduce_tensor_without_average(inp):
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def load_crop_info_dataset(path):
    dataset = dict()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(' ')
            dataset[item[0]] = [int(i) for i in item[1:]]
    return dataset


def get_confusion_matrix(pred_logit, raw_label, num_classes, thresh_list):
    confusion_matrix = np.zeros((len(thresh_list), num_classes, 4))

    for t, thresh in enumerate(thresh_list):
        for i in range(0, num_classes):
            pred = pred_logit[i] > thresh
            label = (raw_label == i + 1)
            tp = np.sum(label & pred)
            p = np.sum(pred)
            fn = np.sum(label) - tp
            confusion_matrix[t, i] += np.array((tp, p, fn, 1))

    return confusion_matrix


def parse_confusion_matrix(confusion_matrix, thresh_list, nan_to_num=None):
    tp_list = confusion_matrix[..., 0]
    p_list = confusion_matrix[..., 1]
    fn_list = confusion_matrix[..., 2]
    # num_list = confusion_matrix[..., 3] # for debug

    if len(thresh_list) > 1:
        index = int(np.argmax(thresh_list == 0.5))
    else:
        index = 0

    tp = tp_list[index]
    p = p_list[index]
    fn = fn_list[index]

    ppv = tp / p
    s = tp / (tp + fn)
    f1 = 2 * tp / (p + tp + fn)
    iou = tp / (p + fn)

    num_classes = confusion_matrix.shape[1]
    aupr = np.zeros((num_classes,), dtype=np.float)

    ppv_list = np.nan_to_num(tp_list / p_list, nan=1)
    s_list = np.nan_to_num(tp_list / (tp_list + fn_list), nan=0)

    for i in range(0, len(aupr)):
        x = s_list[:, i]
        y = ppv_list[:, i]
        aupr[i] = auc(x, y)

    if nan_to_num is not None:
        return np.nan_to_num(iou, nan=nan_to_num), \
               np.nan_to_num(f1, nan=nan_to_num), \
               np.nan_to_num(ppv, nan=nan_to_num), \
               np.nan_to_num(s, nan=nan_to_num), \
               np.nan_to_num(aupr, nan=nan_to_num)
    else:
        return iou, f1, ppv, s, aupr


def summary_result(results, num_classes=4, class_names=None):
    eval_results = {}

    iou, f1, ppv, s, aupr = results
    summary_str = '\n '

    line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
    summary_str += line_format.format('Class', 'IoU', 'F1', 'PPV', 'S', 'AUPR')
    for i in range(num_classes):
        ppv_str = '{:.2f}'.format(ppv[i] * 100)
        s_str = '{:.2f}'.format(s[i] * 100)
        f1_str = '{:.2f}'.format(f1[i] * 100)
        iou_str = '{:.2f}'.format(iou[i] * 100)
        aupr_str = '{:.2f}'.format(aupr[i] * 100)
        summary_str += line_format.format(class_names[i], iou_str, f1_str, ppv_str, s_str, aupr_str)

    mIoU = np.nanmean(np.nan_to_num(iou[-4:], nan=0))
    mF1 = np.nanmean(np.nan_to_num(f1[-4:], nan=0))
    mPPV = np.nanmean(np.nan_to_num(ppv[-4:], nan=0))
    mS = np.nanmean(np.nan_to_num(s[-4:], nan=0))
    mAUPR = np.nanmean(np.nan_to_num(aupr[-4:], nan=0))

    summary_str += 'Summary:\n'
    line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
    summary_str += line_format.format('Scope', 'mIoU', 'mF1', 'mPPV', 'mS', 'mAUPR')

    iou_str = '{:.2f}'.format(mIoU * 100)
    f1_str = '{:.2f}'.format(mF1 * 100)
    ppv_str = '{:.2f}'.format(mPPV * 100)
    s_str = '{:.2f}'.format(mS * 100)
    aupr_str = '{:.2f}'.format(mAUPR * 100)
    summary_str += line_format.format('global', iou_str, f1_str, ppv_str, s_str, aupr_str)

    eval_results['mIoU'] = mIoU
    eval_results['mF1'] = mF1
    eval_results['mPPV'] = mPPV
    eval_results['mS'] = mS
    eval_results['mAUPR'] = mAUPR

    return eval_results, summary_str


def evaluate(config, testloader, model, test_dataset, writer_dict=None):
    model.eval()
    ave_loss = AverageMeter()
    thresh_list = np.linspace(0, 1, 11)  # 0.1

    num_classes = config.DATASET.NUM_CLASSES
    confusion_matrix = np.zeros((len(thresh_list), num_classes, 4))

    crop_info_dataset = None
    if config.DATASET.TEST_CROP_INFO_PATH:
        crop_info_dataset = load_crop_info_dataset(config.DATASET.TEST_CROP_INFO_PATH)

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            idx = dist.get_world_size() * i + dist.get_rank()

            image, label, image_size, image_name = batch
            image = image.cuda()
            label = label.long().cuda()

            if crop_info_dataset is None:
                losses, x = model(image, label)
            else:
                top, bottom, left, right, ori_h, ori_w = crop_info_dataset[image_name[0]]
                losses, x = model(image, label[:, top:(ori_h - bottom), left:(ori_w - right)])

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

            if idx >= len(test_dataset):
                continue

            size = (image_size[0][0].item(), image_size[0][1].item())
            x = F.interpolate(input=x, size=size, mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            pred = torch.sigmoid(x).detach().cpu().numpy()[0]

            if crop_info_dataset:
                top, bottom, left, right, ori_h, ori_w = crop_info_dataset[image_name[0]]
                pred = np.pad(pred, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)

            label = label.cpu().numpy()
            confusion_matrix += get_confusion_matrix(pred, label, num_classes, thresh_list)

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor_without_average(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    avg_loss = ave_loss.average()

    results = parse_confusion_matrix(confusion_matrix, thresh_list)
    # iou, f1, ppv, s, aupr = results  # results for every class
    avg_results, result_str = summary_result(results, num_classes=4, class_names=config.DATASET.CLASS_NAMES)

    if writer_dict is not None:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']

        writer.add_scalar('mIoU', avg_results['mIoU'], global_steps)
        writer.add_scalar('mF1', avg_results['mF1'], global_steps)
        writer.add_scalar('mPPV', avg_results['mPPV'], global_steps)
        writer.add_scalar('mS', avg_results['mS'], global_steps)
        writer.add_scalar('mAUPR', avg_results['mAUPR'], global_steps)

        writer.add_scalar('valid_loss', avg_loss, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return avg_loss, results, result_str
