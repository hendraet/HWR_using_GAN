from pathlib import Path


def create_train_partition(writer, img_path):
    label_file_path = f'HWR_train_partitions/partition_{writer}'
    write_label_file(img_path, label_file_path)


def create_train_partition_for_run(run_id, img_path):
    label_file_path = f'HWR_train_partitions/run_partitions/partition_{run_id}'
    write_label_file(img_path, label_file_path)


def write_label_file(img_path, label_file_path):
    with open(label_file_path, 'w+') as label_file:
        for img in Path(img_path).iterdir():
            filename = img.name
            label = filename.split('.')[1].split('-')[0]
            line = f'{filename} {label}'
            label_file.write(line)


