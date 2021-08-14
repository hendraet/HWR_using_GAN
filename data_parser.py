# TODO Move hard-coded paths


def parse_data(writer):

    image_files, labels = [], []
    with open('data/img_writer_mapping.txt', 'r') as writer_mapping:
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
