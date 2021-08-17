import yaml
from os import listdir

with open('config.yaml') as f:
    config = yaml.safe_load(f)


def parse_data(writer):
    image_files, labels = [], []
    with open(config['iam_writer_mappings'], 'r') as writer_mapping:
        for line in writer_mapping.readlines():
            if line.split(',')[0] == writer:
                labels.append(" ".join(line.split(' ')[1:]))
                image_files.append(line.split(',')[1].split(' ')[0])

    with open(f'train_images_names/style_of_{writer}', 'w+') as f:
        for index, (image_file, label) in enumerate(zip(image_files, labels)):
            line = f'{writer},{image_file} {label}'
            f.write(line)
            if index >= 14:
                break

    with open(f'HWR_Groundtruth/gt_{writer}', 'w+') as f:
        for image_file, label in zip(image_files, labels):
            line = f'{image_file}.png {label}'
            f.write(line)


def create_writer_id(synthesized_img_folder):
    file_names = [int(folder) for folder in listdir(synthesized_img_folder)]
    if not len(file_names) == 0:
        run_id = max(file_names) + 1
    else:
        run_id = 1
    return run_id
