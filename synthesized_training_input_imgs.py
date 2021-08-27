from create_train_part import create_train_partition
from GAN.create_imgs import create_images_from_source
from HWR.train_with_synth_imgs import train_with_synth_imgs_from_input_folder, test_with_imgs_from_input_folder
from pathlib import Path
import argparse
import yaml
from data_parser import create_writer_id

parser = argparse.ArgumentParser(description='Create synthesized images from input images and train HWR with them.')
parser.add_argument('--n-generated-images', default=150, type=int,
                    help='The number of images that the GAN will produce and the HWR train on.')
parser.add_argument('--input_folder', default='washington_input/', help='Folder that contains the input images.')
parser.add_argument('-t', action='store_true', help='If specified, the model will be tested on images in the test_folder and compared to the default model.')

args = parser.parse_args()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

synthesized_img_folder = Path(config['result_paths']['synthesized_images'], 'runs')
run_id = str(create_writer_id(synthesized_img_folder))

# Synthesized images are saved in respective run_id folder.
create_images_from_source(
    run_id,
    args.n_generated_images,
    args.input_folder,
    config['gan_default_model'])
hwr_training_labels_file = create_train_partition(run_id, synthesized_img_folder / run_id, 'runs')
trained_model_path = train_with_synth_imgs_from_input_folder(run_id, hwr_training_labels_file, synthesized_img_folder / run_id)


if args.t:
    test_folder = Path('washington_test/')
    labels_file = test_folder / 'labels.txt'
    test_with_imgs_from_input_folder(labels_file, test_folder,
                                     trained_model_path, f'id_{run_id}_')
    test_with_imgs_from_input_folder(labels_file, test_folder, config['hwr_default_model'],
                                     f'id_{run_id}_original_')
