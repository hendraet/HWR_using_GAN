import pandas as pd
import sys


with open('HWR/RWTH_partition/RWTH.iam_word_gt_final.test.thresh', 'r') as f_ocr:
    data_ocr = f_ocr.readlines()
    file_label_all = [i[:-1].split(' ') for i in data_ocr]
    file_label_img_writer = [i[0].split(',') for i in file_label_all]
    ocr_list = [[file_label_img_writer[i][0], file_label_img_writer[i][1], file_label_all[i][1]]
        for i, line in enumerate(file_label_img_writer)]

with open('GAN/Groundtruth/gan.iam.test.gt.filter27', 'r') as f_gan:

    gan_test_data = f_gan.readlines()
    file_label_all = [i[:-1].split(' ') for i in gan_test_data]
    file_label_img_writer = [i[0].split(',') for i in file_label_all]
    gan_list = [[file_label_img_writer[i][1], file_label_img_writer[i][0], file_label_all[i][1]]
        for i, line in enumerate(file_label_img_writer)]



df_ocr = pd.DataFrame(ocr_list)
df_gan = pd.DataFrame(gan_list)

w_ocr = set(df_ocr[1].values)
w_gan = set(df_gan[1].values)
#writers = w_ocr.intersection(w_gan)
writer = sys.argv[1]
# writers = df_ocr[df_ocr[1].isin(w_intersect)]

# histogram_ocr = df_ocr.groupby(1).count().sort_values(0, ascending=False)
# histogram_gan = df_gan.groupby(1).count().sort_values(0, ascending=False)

# histogram_subset = pd.merge(histogram_gan, histogram_ocr, how='inner', on=[0])
# writers = histogram_subset[0:9].index.values.tolist()

images = df_gan[df_gan[1] == str(writer)]

with open(f'train_images_names/style_of_{writer}', 'w+') as f:
    i = 0
    for index, image in images.iterrows():
        tmp = image[1] + ',' + image[0] + ' ' + image[2] + '\n'
        f.write(tmp)
        i += 1
        if i > 14:
            break
    f.close()

with open(f'HWR_Groundtruth/gt_{writer}', 'w') as f:
    i = 0
    for index, image in images.iterrows():
        tmp = image[0] + ',' + image[1] + ' ' + image[2] + '\n'
        f.write(tmp)
    f.close()
