import os
import sys

writer = sys.argv[1]
path = '/home/padl21t1/research-GANwriting/Synthesized_data/'+writer+'/'

train_partition = open('train_OCR_'+writer+'.part','w+')

for f in os.listdir(path):
    writer = f.split('-')[0]
    filename = f[:-4]
    label = f.split('.')[1].split('-')[0]
    s = filename + "," + writer + " " + label + "\n"
    train_partition.write(s)

train_partition.close()

