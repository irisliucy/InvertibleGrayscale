
'''
compute PSNR with tensorflow
'''
import tensorflow as tf
import csv
import sys

from config import *
from model import *

psnr_result = []

def write_to_csv(path, data_dict):
    ''' write result to csv
    Args:
        path (string): file directory
        data_dict (dict): dictionary of data {header: value}
    '''
    exists_or_mkdir(RESULT_CSV_DIR)
    with open(path, 'w') as f:
        fieldnames = data_dict.keys()
        file = csv.DictWriter(f, fieldnames = fieldnames)

        file.writeheader() # write header
        file.writerow(data_dict)

def read_img(path):
	return tf.image.decode_image(tf.read_file(path))

def psnr(tf_img1, tf_img2):
    for i in range(len(tf_img1)):
        psnr_val = tf.image.psnr(tf_img1[i], tf_img2[i], max_val=255)
        psnr_result.append(psnr_val)
    return psnr_result

def main():
    src_image_list = glob.glob(os.path.join(SOURCE_EVAL_DIR, '*.*'))
    target_image_list =  glob.glob(os.path.join(TARGET_EVAL_DIR, '*.*'))

    # Check if lists are ordered
    # for i in range(3):
    #     print(sorted(src_image_list)[i], '\n', sorted(target_image_list)[i])

    print("Preparing images....")
    t1 = [read_img(x) for x in sorted(src_image_list)]
    t2 = [read_img(x) for x in sorted(target_image_list)]
    print("source images num: {} \ntarget images num: {}".format(len(t1), len(t2)))

    if len(t1) != len(t2):
        print("Image lists not comparable! Source list contains {} images and target list contains {} images".format(len(t1), len(t2)))
        sys.exit()

    print("Computing PSNR....")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        psnr_results = sess.run(psnr(t1, t2))
        average_psnr = sum(psnr_results) / len(psnr_results)
        print("PSNR: {}".format(average_psnr))
        # write result to csv
        write_to_csv(os.path.join(RESULT_CSV_DIR, 'evaluated_result.csv'),
                                    {'PSNR': average_psnr})
        print('Evaluation Completed!')

if __name__ == '__main__':
    print("Evaluation starts.")
    main()
