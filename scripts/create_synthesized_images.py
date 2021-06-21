import sys
import subprocess

writer = sys.argv[1]


# create index file with words for writer in train_images_names
# create inde file with words for HWR testing in HWR_Groundtruth
subprocess.call(['python3', 'data-parser.py', writer])

# create images
subprocess.call(['python3', '../GAN/tt.test_single_writer.4_scenarios.py', writer], cwd='../GAN/')

# create HWR training partition from generated words
subprocess.call(['python3', 'create_train_part.py', writer])

