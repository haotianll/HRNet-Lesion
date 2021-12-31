import argparse
from collections import OrderedDict

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='~/userfolder/mmsegmentation-lesion/work_dirs/xxx/iter_40000.pth')
    parser.add_argument('output', default='~/userfolder/HRNet-Semantic-Segmentation/temp.pth')
    args = parser.parse_args()

    key_dict = {
        'decode_head.convs.0.conv.': 'model.last_layer.0.',
        'decode_head.convs.0.bn.': 'model.last_layer.1.',
        'decode_head.conv_seg.': 'model.last_layer.3.',
        'backbone.': 'model.'
    }

    new_pth = OrderedDict()
    pth = torch.load(args.input, map_location=torch.device('cpu'))['state_dict']
    for key, weight in pth.items():
        new_key = key
        for k, v in key_dict.items():
            new_key = new_key.replace(k, v)
        new_pth[new_key] = weight

        print(key, '->', new_key)

    torch.save(new_pth, args.output)
