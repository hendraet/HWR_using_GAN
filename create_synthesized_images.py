import sys
import os
import shutil
import config
import argparse

from GAN import create_imgs_from_IAM
from HWR import *
from GAN import *
from HWR import train_with_synth_imgs
from create_train_part import create_train_partition
from data_parser import parse_data

parser = argparse.ArgumentParser(description='create synthesized pics and train them.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('writer', type=int, help='add the id of a writer from the IAM Dataset.')
parser.add_argument('n_generated_images', type=int,
                    help='The number of images that the GAN will produce and the HWR train on.')
args = parser.parse_args()

writer = args.writer
n_generated_images = args.n_generated_images

# create index file with words for writer in train_images_names
# create index file with words for HWR testing in HWR_Groundtruth
#subprocess.call(['python3', 'data_parser.py', writer])
parse_data(writer)

# delete old images
filepath = f'synthesized_images/{writer}'
if os.path.isdir(filepath):
    shutil.rmtree(filepath)
#subprocess.call(f'rm synthesized_images/{writer}/*', shell=True)

# create images
#subprocess.call(['python3', 'create_imgs_from_IAM.py', writer, n_generated_images], cwd='GAN/')
create_imgs_from_IAM.create_images_of_writer(writer, n_generated_images, config.data_dir)

# create HWR training partition from generated words

create_train_partition(writer, f'synthesized_images/{writer}/')
# subprocess.call(['python3', 'create_train_part.py', writer])


# train HWR with synthesized images of this writer
train_with_synth_imgs.train_with_synthesized_imgs(writer, n_generated_images)