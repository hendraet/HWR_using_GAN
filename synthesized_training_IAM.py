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
synthesized_img_path = f'synthesized_images/{writer}/'

# create index file with words for writer in train_images_names
# create index file with words for HWR testing in HWR_Groundtruth
parse_data(writer)

# delete old images
if os.path.isdir(synthesized_img_path):
    shutil.rmtree(synthesized_img_path)

# create images
create_imgs_from_IAM.create_images_of_writer(writer, n_generated_images, config.data_dir)

# create HWR training partition from generated words
create_train_partition(writer, synthesized_img_path)

# train HWR with synthesized images of this writer
labels_file = f'HWR_train_partitions/partition_{writer}'
train_with_synth_imgs.train_and_test_with_synthesized_imgs(writer, n_generated_images, labels_file, synthesized_img_path)
