import os
import sys

# writer = int(sys.argv[1])
# if writer < 1000:
#     path = 'synthesized_images/'+writer+'/'
# else :


def create_train_partition(writer, path):

    train_partition = open(f'HWR_train_partitions/partition_{writer}', 'w+')

    for f in os.listdir(path):
        writer = f.split('-')[0]
        filename = f[:-4]
        label = f.split('.')[1].split('-')[0]
        s = filename + "," + writer + " " + label + "\n"
        train_partition.write(s)

    train_partition.close()


def create_train_partition_for_run(run_id, path):
    train_partition = open(f'HWR_train_partitions/run_partitions/partition_{run_id}', 'w+')

    for f in os.listdir(path):
        writer = f.split('-')[0]
        filename = f[:-4]
        label = f.split('.')[1].split('-')[0]
        s = filename + "," + writer + " " + label + "\n"
        train_partition.write(s)

    train_partition.close()

