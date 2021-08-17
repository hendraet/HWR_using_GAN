from pathlib import Path


def create_train_partition(writer, img_path, labels_path, dataset='iam'):
    label_file_path = Path(labels_path, dataset, f'train_partition_for_{writer}')
    label_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_file_path, 'w+') as label_file:
        for img in Path(img_path).iterdir():
            filename = img.name
            label = filename.split('.')[1].split('-')[0]
            line = f'{filename} {label}\n'
            label_file.write(line)
    return label_file_path
