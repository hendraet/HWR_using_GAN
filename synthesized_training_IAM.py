import shutil
import yaml
import argparse
from pathlib import Path
from GAN.create_imgs import create_images_from_source
from HWR.train_with_synth_imgs import train_and_test_with_synthesized_imgs
from create_train_part import create_train_partition
from data_parser import parse_data

# TODO Merge into main script and add --iam flag
parser = argparse.ArgumentParser(
    description='Create synthesized images from the IAM dataset and train the HWR with them.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--writer',
    type=str,
    help='Specify which IAM writer should be trained.')
parser.add_argument('--n-generated-images', type=int, default=150,
                    help='The number of images that the GAN will produce and the HWR train on.')
args = parser.parse_args()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

writer = args.writer
synthesized_img_path = Path(
    config["result_paths"]["synthesized_images"], 'iam', writer)

# create index file with words of this writer in train_images_names for creating imgs with GAN
# create index file with words for HWR testing in HWR_Groundtruth
parse_data(writer)

# delete old images
if synthesized_img_path.exists():
    shutil.rmtree(synthesized_img_path)

# create images
create_images_from_source(
    writer,
    args.n_generated_images,
    config['data_dir'],
    config['gan_default_model'],
    'iam')

# create HWR training partition from generated words
hwr_training_labels_file = create_train_partition(writer, synthesized_img_path, config['result_paths']['labels_path'])
# train HWR with synthesized images of this writer
train_and_test_with_synthesized_imgs(writer, args.n_generated_images, hwr_training_labels_file, synthesized_img_path)

# TODO save model
