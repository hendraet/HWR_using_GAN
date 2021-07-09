import sys
import config
from create_train_part import create_train_partition_for_run
from GAN.create_imgs_from_IAM import create_images_from_input_folder
import argparse
sys.path.append('./HWR')
from os import listdir
from HWR.train_with_synth_imgs import train_with_synth_imgs_from_input_folder, test_with_imgs_from_input_folder

parser = argparse.ArgumentParser(description='create synthesized pics from input imgs and train HWR with them.')
parser.add_argument('--n_generated_images', default=150, type=int)
args = parser.parse_args()

n_generated_images = args.n_generated_images
synthesized_img_folder = 'synthesized_images/run'
hwr_training_folder = 'HWR_train_partitions/run_partitions'

# find run_id
file_names = [int(f) for f in listdir(synthesized_img_folder)]
if not len(file_names) == 0:
    run_id = max(file_names) + 1
else:
    run_id = 1

# create images
create_images_from_input_folder(run_id, n_generated_images, 'input_imgs/')

# create HWR training partition from generated words
create_train_partition_for_run(run_id, f'{synthesized_img_folder}/{run_id}')

# train HWR with synthesized images of this writer
labels_file = f'HWR_train_partitions/run_partitions/partition_{run_id}'
result_folder = f'results/{run_id}'
train_with_synth_imgs_from_input_folder(run_id, labels_file, f'{synthesized_img_folder}/{run_id}/')

# test trained model on test imgs
labels_file = f'test_imgs/labels.txt'
test_with_imgs_from_input_folder(labels_file, 'test_imgs/', f'final_weights_HWR/{run_id}/seq2seq-{run_id}.model', f'id_{run_id}_')

# compare to baseline model
test_with_imgs_from_input_folder(labels_file, 'test_imgs/', config.hwr_default_model, f'id_{run_id}_original_')
