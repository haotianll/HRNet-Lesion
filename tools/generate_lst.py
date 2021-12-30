import argparse
import os

image_dir = 'image'
label_dir = 'label'
splits = ['train', 'val', 'test']
image_dirs = [
    'image/{}',
    'image/{}_crop'
]
label_dirs = [
    'label/{}/annotations',
    'label/{}/annotations_crop',
]


def generate(root):
    assert len(image_dirs) == len(label_dirs)

    for split in splits:
        for image_path, label_path in zip(image_dirs, label_dirs):
            image_path = image_path.format(split)
            label_path = label_path.format(split)

            if split != 'train' and image_path.endswith('_crop'):
                label_path = label_path.replace('_crop', '')

            if not os.path.exists(os.path.join(root, label_path)):
                continue

            lines = []
            for label in os.listdir(os.path.join(root, label_path)):
                image = label.replace('.png', '.jpg')

                if os.path.exists(os.path.join(root, image_path, image)):
                    lines.append('{} {}\n'.format(os.path.join(image_path, image), os.path.join(label_path, label)))
                else:
                    print('not found: {}'.format(os.path.join(root, image_path, image)))

            print(image_path, label_path, len(lines))

            output_file = '{}.lst'.format(image_path.split('/')[1])
            with open(os.path.join(root, output_file), 'w') as f:
                f.writelines(lines)

            print(f'Save to {os.path.join(root, output_file)}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='path of dataset root')
    args = parser.parse_args()

    generate(args.root)
