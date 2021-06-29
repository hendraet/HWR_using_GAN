import os
import sys

writer = sys.argv[1]
path = '../synthesized_images/'+writer+'/'

train_partition = open(f'../HWR_train_partitions/partition_{writer}','w+')

for f in os.listdir(path):
    writer = f.split('-')[0]
    filename = f[:-4]
    label = f.split('.')[1].split('-')[0]
    s = filename + "," + writer + " " + label + "\n"
    train_partition.write(s)

train_partition.close()

