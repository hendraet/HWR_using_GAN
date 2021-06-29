import sys
sys.path.append('./HWR')
import subprocess
from HWR.train_with_synth_imgs import train_with_synthesized_imgs
from create_train_part import create_train_partition

writer = sys.argv[1]
n_generated_images = sys.argv[2]

# create index file with words for writer in train_images_names
# create index file with words for HWR testing in HWR_Groundtruth
subprocess.call(['python3', 'data-parser.py', writer])

# delete old images
subprocess.call(f'rm synthesized_images/{writer}/*', shell=True)
# create images
subprocess.call(['python3', 'tt.test_single_writer.4_scenarios.py', writer, n_generated_images], cwd='GAN/')

# create HWR training partition from generated words
create_train_partition(writer, 'synthesized_images/'+writer+'/')
# subprocess.call(['python3', 'create_train_part.py', writer])

# train HWR with synthesized images of this writer
train_with_synthesized_imgs(writer, n_generated_images)