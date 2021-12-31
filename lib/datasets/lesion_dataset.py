import os

import cv2
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset


class LesionDataset(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=4,
                 multi_scale=True,
                 flip=True,
                 rotate=True,
                 base_size=1440,
                 image_scale=(1200, 1440),
                 ratio_range=(0.5, 2.0),
                 crop_size=(1200, 1440),
                 pad_size=None,
                 ignore_label=-1,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[116.513, 56.437, 16.309],
                 std=[80.206, 41.232, 13.293],
                 split='train'):

        super(LesionDataset, self).__init__(ignore_label, base_size,
                                            crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()
        self.split = split
        if num_samples:
            self.files = self.files[:num_samples]

        # NEW:
        self.image_scale = image_scale
        self.ratio_range = ratio_range
        self.crop_size = crop_size

        if pad_size is None and self.crop_size is not None:
            self.pad_size = self.crop_size  # CHECK THIS
        else:
            self.pad_size = pad_size

        self.multi_scale = multi_scale
        self.flip = flip
        self.rotate = rotate

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                'img': image_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        size = image.shape

        label = np.array(Image.open(os.path.join(self.root, item["label"])).convert('P'))

        info_dict = {
            'filename': item['name'],
            'image_path': os.path.join(self.root, item['img']),
            'label_path': os.path.join(self.root, item['label']),
            'origin_shape': image.shape
        }
        if self.split != 'train':
            image = self.random_scale(image, label=None, ratio_range=None)
            image = self.normalize(image, self.mean, self.std, to_rgb=True)
            info_dict['input_shape'] = image.shape

            if self.pad_size is not None:
                h, w = image.shape[:2]
                image = self.pad_image(image, h, w, self.pad_size, (0,))
                info_dict['pad_shape'] = image.shape
            else:
                info_dict['pad_shape'] = None

            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name, info_dict

        image, label, info = self.generate_sample(image, label)
        info_dict['input_shape'], info_dict['pad_shape'] = info
        return image.copy(), label.copy(), np.array(size), name, info_dict

    def random_scale(self, image, label=None, ratio_range=None):
        if ratio_range is not None:
            ratio = np.random.random_sample() * (ratio_range[1] - ratio_range[0]) + 0.5
        else:
            ratio = 1.

        h, w = image.shape[:2]
        scale = int(self.image_scale[0] * ratio), int(self.image_scale[1] * ratio)
        scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
        new_w = int(w * float(scale_factor) + 0.5)
        new_h = int(h * float(scale_factor) + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def random_crop(self, image, label, crop_size, cat_max_ratio=1.):
        def get_crop_bbox(img, crop_size):
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        def crop(img, crop_bbox):
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            return img

        crop_bbox = get_crop_bbox(image, crop_size)
        if cat_max_ratio < 1.:
            for _ in range(10):
                seg_temp = crop(label, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_label]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                    break
                crop_bbox = get_crop_bbox(image, crop_size)

        image = crop(image, crop_bbox)
        label = crop(label, crop_bbox)

        return image, label

    # horizontal flip
    def random_flip(self, image, label, prob=0.5):
        if np.random.rand() < prob:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)

        return image, label

    def pad(self, image, label, size, pad_val=0, seg_pad_val=0):
        if size is not None:
            h, w = label.shape
            image = self.pad_image(image, h, w, size, (pad_val,))
            label = self.pad_image(label, h, w, size, (seg_pad_val,))

        return image, label

    def random_rotate(self, image, label,
                      prob=1.0, degree=(-45, 45), pad_val=0, seg_pad_val=0,
                      center=None, auto_bound=False):
        # REF: mmcv.imrotate
        def imrotate(img,
                     angle,
                     center=None,
                     scale=1.0,
                     border_value=0,
                     auto_bound=False,
                     interpolation=cv2.INTER_LINEAR):

            if center is not None and auto_bound:
                raise ValueError('`auto_bound` conflicts with `center`')
            h, w = img.shape[:2]
            if center is None:
                center = ((w - 1) * 0.5, (h - 1) * 0.5)
            assert isinstance(center, tuple)

            matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            if auto_bound:
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                new_w = h * sin + w * cos
                new_h = h * cos + w * sin
                matrix[0, 2] += (new_w - w) * 0.5
                matrix[1, 2] += (new_h - h) * 0.5
                w = int(np.round(new_w))
                h = int(np.round(new_h))
            rotated = cv2.warpAffine(img, matrix, (w, h), flags=interpolation, borderValue=border_value)
            return rotated

        rotate = True if np.random.rand() < prob else False
        if rotate:
            degree = np.random.uniform(min(*degree), max(*degree))
            image = imrotate(
                image,
                angle=degree,
                border_value=pad_val,
                center=center,
                auto_bound=auto_bound,
                interpolation=cv2.INTER_LINEAR)

            label = imrotate(
                label,
                angle=degree,
                border_value=seg_pad_val,
                center=center,
                auto_bound=auto_bound,
                interpolation=cv2.INTER_NEAREST)
        return image, label

    def normalize(self, img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def generate_sample(self, image, label):
        if self.multi_scale:
            # 'Resize' ratio_range=(0.5, 2.0)
            image, label = self.random_scale(image, label, ratio_range=self.ratio_range)

            # 'RandomCrop' crop_size=(1200, 1440)
            image, label = self.random_crop(image, label, crop_size=self.crop_size, cat_max_ratio=0.75)

        # 'RandomFlip', prob=0.5, direction='horizontal'
        if self.flip:
            image, label = self.random_flip(image, label, prob=0.5)

        # 'RandomRotate', prob=1.0, pad_val=0, seg_pad_val=0, degree=(-45, 45), auto_bound=False
        if self.rotate:
            image, label = self.random_rotate(image, label, prob=1.0, degree=(-45, 45), pad_val=0, seg_pad_val=0)

        image = self.normalize(image, self.mean, self.std, to_rgb=True)
        label = self.label_transform(label)

        input_shape = image.shape

        # 'Pad', size=(1200, 1440), pad_val=0, seg_pad_val=0
        image, label = self.pad(image, label, self.pad_size, pad_val=0, seg_pad_val=0)
        pad_shape = image.shape

        image = image.transpose((2, 0, 1))

        if self.downsample_rate != 1:
            label = cv2.resize(label, None, fx=self.downsample_rate, fy=self.downsample_rate,
                               interpolation=cv2.INTER_NEAREST)

        return image, label, (input_shape, pad_shape)
