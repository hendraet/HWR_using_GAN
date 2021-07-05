import sys

from create_train_part import create_train_partition_for_run
from GAN.create_imgs_from_IAM import  create_images_from_input_folder

sys.path.append('./HWR')
import subprocess
from os import listdir
from HWR.train_with_synth_imgs import train_with_synth_imgs_from_input_folder

n_generated_images = 10
if sys.argv[1]:
    n_generated_images = int(sys.argv[1])

synthesized_img_folder = 'synthesized_images/run/'
hwr_training_folder = 'HWR_train_partitions/run_partitions'


# find run_id
file_names = [int(f) for f in listdir(synthesized_img_folder)]
if not len(file_names) == 0:
    run_id = max(file_names) + 1
else:
    run_id = 1

# create images
create_images_from_input_folder(run_id, n_generated_images, 'input_imgs/')
# subprocess.call(['python3', 'test_random_imgs_scenario.py', n_generated_images], cwd='GAN/')

# create HWR training partition from generated words
create_train_partition_for_run(run_id, f'{synthesized_img_folder}/{run_id}')

# train HWR with synthesized images of this writer
# TODO work on this
labels_file = f'HWR_train_partitions/run_partitions/partition_{run_id}'
result_folder = f'results/{run_id}'
train_with_synth_imgs_from_input_folder(run_id, labels_file, f'{synthesized_img_folder}/{run_id}/')
