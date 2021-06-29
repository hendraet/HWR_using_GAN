import sys

from create_train_part import create_train_partition

sys.path.append('./HWR')
import subprocess
from os import listdir
from HWR.train_with_synth_imgs import train_with_synthesized_imgs

n_generated_images = sys.argv[1]


# create images
subprocess.call(['python3', 'test_random_imgs_scenario.py', n_generated_images], cwd='GAN/')

# create HWR training partition from generated words
folder = 'synthesized_images/unknown_authors/'
run_id = max([int(f) for f in listdir(folder)])
create_train_partition(run_id, 'folder'+run_id+'/')

# train HWR with synthesized images of this writer
# TODO work on this
# train_with_synthesized_imgs(writer, n_generated_images)