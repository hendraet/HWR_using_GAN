from create_train_part import create_train_partition_for_run
from GAN.create_imgs_from_IAM import create_images_from_input_folder
from HWR.train_with_synth_imgs import train_with_synth_imgs_from_input_folder, test_with_imgs_from_input_folder
import argparse
from os import listdir

# TODO remove this

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create synthesized images from input images and train HWR with them.')
    parser.add_argument('--n-generated-images', default=150, type=int,
                        help='The number of images that the GAN will produce and the HWR train on.')
    parser.add_argument('--model-id', default=3050, type=int, help='Model filename for GAN')
    args = parser.parse_args()

    # TODO Move paths to yaml file
    synthesized_img_folder = 'synthesized_images/run'
    hwr_training_folder = 'HWR_train_partitions/run_partitions'
    model = f'/home/padl21t1/research-GANwriting/save_weights/contran-{args.model_id}.model'

    # TODO Move to util function
    file_names = [int(f) for f in listdir(synthesized_img_folder)]
    if not len(file_names) == 0:
        run_id = max(file_names) + 1
    else:
        run_id = 1

    labels_file = f'HWR_train_partitions/run_partitions/partition_{run_id}'
    result_folder = f'results/{run_id}'

    # TODO hard-coded path
    # Synthesized images are saved in respective run_id folder.
    # Then, corresponding partition file for the HWR is created.
    # TODO: In yaml specified HWR model ist used for training of new writer.
    create_images_from_input_folder(run_id, args.n_generated_images, 'washington_input/', model)
    create_train_partition_for_run(run_id, f'{synthesized_img_folder}/{run_id}')
    train_with_synth_imgs_from_input_folder(run_id, labels_file, f'{synthesized_img_folder}/{run_id}/')

    # TODO hard-coded path
    # TODO: flag for testing / comparing?
    # Trained HWR can be tested with original input images
    # and compared to previous writer-untrained HWR.
    labels_file = f'washington_test/labels.txt'
    test_with_imgs_from_input_folder(labels_file, 'washington_test/',
                                     f'final_weights_HWR/{run_id}/seq2seq-{run_id}.model', f'id_{run_id}_')
    test_with_imgs_from_input_folder(labels_file, 'washington_test/', config.hwr_default_model,
                                     f'id_{run_id}_original_')
