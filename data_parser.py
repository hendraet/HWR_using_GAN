import pandas as pd


def parse_data(writer):

    with open('GAN/Groundtruth/gan.iam.test.gt.filter27', 'r') as f_gan:
        gan_test_data = f_gan.readlines()
        file_label_all = [i[:-1].split(' ') for i in gan_test_data]
        file_label_img_writer = [i[0].split(',') for i in file_label_all]
        gan_list = [[file_label_img_writer[i][1], file_label_img_writer[i][0], file_label_all[i][1]]
                    for i, line in enumerate(file_label_img_writer)]

    df_gan = pd.DataFrame(gan_list)
    images = df_gan[df_gan[1] == str(writer)]

    with open(f'train_images_names/style_of_{writer}', 'w+') as f:
        i = 0
        for index, image in images.iterrows():
            tmp = image[1] + ',' + image[0] + ' ' + image[2] + '\n'
            f.write(tmp)
            i += 1
            if i > 14:
                break

    with open(f'HWR_Groundtruth/gt_{writer}', 'w') as f:
        for index, image in images.iterrows():
            tmp = f'{image[0]}.png {image[2]}\n'
            f.write(tmp)