import os
import numpy as np
import tensorflow as tf
import datetime, time, scipy.io
import cv2
import ray
import ray.tune as tune

from tfrecords import *
from model import *
from util import *
from config import *

# tf.logging.set_verbosity(tf.logging.ERROR) # suppress annoying tf warnings


# --------------------------------- HYPER-PARAMETERS --------------------------------- #
in_channels = 3
out_channels = 3
n_epochs1 = 90
n_epochs2 = 30
batch_size = 4
learning_rate = 0.0002
beta1 = 0.9

display_steps = 100
save_epochs = 10

def gen_list(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.*'))
    file_list.sort()
    file_pair_list = []
    for i, path1 in enumerate(file_list):
        if SAMPLE_TEST_MODE==True:
            if i < NUMBER_OF_SAMPLES:
                from PIL import Image
                img = Image.open(path1)
                img = img.resize(IMG_SHAPE, Image.ANTIALIAS)
                img.save(path1)
                file_pair_list.append(path1)
        else:
            file_pair_list.append(path1)
    return file_pair_list


def train(train_list, val_list, debug_mode=DEBUG_MODE, config=None, reporter=None):
    print('Running ColorEncoder -Training!')
    print('Noise Mode (add noise to training): ', NOISE_MODE)
    init_start_time = time.time()
    # create folders to save trained model and results
    checkpt_dir = os.path.join(RESULT_STORAGE_DIR, 'checkpoints')
    record_dir = os.path.join(RESULT_OUTPUT_DIR, 'tfrecords')
    result_loss_dir = os.path.join(RESULT_OUTPUT_DIR, 'loss')
    result_imgs_dir = os.path.join(RESULT_OUTPUT_DIR, 'train_result_imgs')
    exists_or_mkdir(RESULT_GRAPH_DIR, need_remove=True)
    exists_or_mkdir(RESULT_OUTPUT_DIR)
    exists_or_mkdir(checkpt_dir)
    exists_or_mkdir(record_dir)
    exists_or_mkdir(result_loss_dir, need_remove=False)
    exists_or_mkdir(result_imgs_dir, need_remove=True)


    # --------------------------------- load data ---------------------------------
    train_num = NUMBER_OF_SAMPLES if SAMPLE_TEST_MODE else len(train_list)
    valid_num = NUMBER_OF_SAMPLES if SAMPLE_TEST_MODE else len(val_list)
    print("Train images:{} \nTest images:{} \nTotal images:{}".format(train_num, valid_num, (train_num + valid_num)))

    # data fetched at range: [-1,1]
    # input_imgs, target_imgs, num = input_producer(train_list, in_channels, batch_size, need_shuffle=True)
    input_imgs, target_imgs, num = tfrecord_input_producer(train_list, record_dir, in_channels, IMG_SHAPE[0], batch_size, need_shuffle=True)

    latent_imgs = encode(input_imgs, 1, is_train=True, reuse=False)
    pred_imgs = decode(latent_imgs, out_channels, is_train=True, reuse=False)
    if debug_mode:
        # input_val, target_val, num_val = input_producer(val_list, in_channels, batch_size, need_shuffle=False)
        input_val, target_val, num_val = tfrecord_input_producer(train_list, record_dir, in_channels, IMG_SHAPE[0], batch_size, need_shuffle=False)
        latent_val = encode(input_val, 1, is_train=False, reuse=True)
        pred_val = decode(latent_val, out_channels, is_train=False, reuse=True)

    # --------------------------------- loss terms ---------------------------------
    with tf.name_scope('Loss'):

        target_224 = tf.image.resize_images(target_imgs, size=[224, 224], method=0, align_corners=False)
        predict_224 = tf.image.resize_images(latent_imgs, size=[224, 224], method=0, align_corners=False)
        vgg19_api = VGG19(os.path.join(CURRENT_DIR, "vgg19.npy"))
        vgg_map_targets = vgg19_api.build((target_224 + 1) / 2, is_rgb=True)
        vgg_map_predict = vgg19_api.build((predict_224 + 1) / 2, is_rgb=False)
        ### greyscale conformity loss
        # stretch the global contrast to follow color contrast
        vgg_loss = 1e-7 * tf.losses.mean_squared_error(vgg_map_targets, vgg_map_predict) # contrast loss
        # suppress local patterns
        gray_inputs = tf.image.rgb_to_grayscale(target_imgs)
        latent_grads = tf.reduce_mean(tf.image.total_variation(latent_imgs)/IMG_SHAPE[0]**2)
        target_grads = tf.reduce_mean(tf.image.total_variation(gray_inputs)/IMG_SHAPE[0]**2)
        grads_loss = tf.abs(latent_grads-target_grads) # local structure loss
        # control the mapping order similar to normal rgb2gray
        global_order_loss = tf.reduce_mean(tf.maximum(70/127.0, tf.abs(gray_inputs-latent_imgs))) - 70/127.0 # lightness loss

        ### quantization loss
        latent_stack = tf.concat([latent_imgs for t in range(IMG_SHAPE[0])], axis=3)
        id_mat = np.ones(shape=(1, 1, 1, 1))
        quant_stack = np.concatenate([id_mat * t for t in range(IMG_SHAPE[0])], axis=3)
        quant_stack = (quant_stack / 127.5) - 1
        quantization_map = tf.reduce_min(tf.abs(latent_stack - quant_stack), axis=3)
        quantization_loss = tf.reduce_mean(quantization_map)

        ### invertibility loss
        mse_loss = tf.losses.mean_squared_error(target_imgs, pred_imgs)

        # add L2 Regularization on each weight
        l2_norms = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        if config != None:
            l2_norm = config['lamda'] * tf.reduce_sum(l2_norms) # tune lambda
        else:
            l2_norm = LAMBDA * tf.reduce_sum(l2_norms)

        ### reconstruction loss
        loss_op1 = 3 * mse_loss + vgg_loss + 0.5*grads_loss + global_order_loss + l2_norm
        loss_op2 = 3 * mse_loss + vgg_loss + 0.1*grads_loss + global_order_loss + 10*quantization_loss + l2_norm

    # --------------------------------- solver definition ---------------------------------
    global_step = tf.Variable(0, name='global_step1', trainable=False)
    iters_per_epoch = np.floor_divide(num, batch_size)
    lr_decay = tf.train.polynomial_decay(learning_rate=learning_rate,
                                          global_step=global_step,
                                          decay_steps=iters_per_epoch*(n_epochs1+n_epochs2),
                                          end_learning_rate=learning_rate / 100.0,
                                          power=0.9)

    with tf.name_scope('optimizer'):
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("encode") or var.name.startswith("decode")]
        train_op1 = tf.train.AdamOptimizer(lr_decay, beta1=beta1).minimize(loss_op1, var_list=gen_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr_decay, beta1=beta1).minimize(loss_op2, var_list=gen_vars, global_step=global_step)

    # --------------------------------- model training ---------------------------------

    with tf.name_scope("parameter_count"):
        num_parameters = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45

    saver = tf.train.Saver(max_to_keep=1)
    total_loss_list = []
    grad_loss_list = []
    vgg_loss_list = []
    order_loss_list = []
    quanti_loss_list = []
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        need_resotre = False
        if need_resotre:
            check_pt = tf.train.get_checkpoint_state("checkpoints")
            if check_pt and check_pt.model_checkpoint_path:
                saver.restore(sess, check_pt.model_checkpoint_path)
                print("pretrained model loaded successfully!")

        print(">>------------>>> [Training_Num] =%d" % num)
        print(">>------------>>> [Parameter_Num] =%d" % sess.run(num_parameters))

        # print("----- Adding noise to the weights of model ...")
        # for weight in [train_op1, loss_op1, grads_loss, vgg_loss, global_order_loss]:
        #     sess.run(add_random_noise(weight))

        # -------------------------------- stage one --------------------------------
        for epoch in range(0, n_epochs1):
            print('---- epoch {}'.format(epoch))
            start_time = time.time()
            epoch_loss, n_iters = 0, 0
            avg_grads, avg_vggs, avg_orders = 0, 0, 0
            for step in range(0, num, batch_size):
                _, loss, grads, vggs, orders = sess.run([train_op1, loss_op1, grads_loss, vgg_loss, global_order_loss])
                epoch_loss += loss
                avg_grads += grads
                avg_vggs += vggs
                avg_orders += orders
                n_iters += 1
                # iteration information
                if n_iters % display_steps == 0:
                    tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print("%s >> [%d/%d] iter: %d  loss: %4.4f" % (tm, epoch, n_epochs1+n_epochs2, n_iters, loss))

            # epoch information
            epoch_loss = epoch_loss / n_iters
            avg_grads = avg_grads / n_iters
            avg_vggs = avg_vggs / n_iters
            avg_orders = avg_orders / n_iters
            total_loss_list.append(epoch_loss)
            grad_loss_list.append(avg_grads)
            vgg_loss_list.append(avg_vggs)
            order_loss_list.append(avg_orders)
            print("[*] ----- Epoch: %d/%d | Loss: %4.4f | Time-consumed: %4.3f -----" %
                  (epoch, n_epochs1+n_epochs2, epoch_loss, (time.time() - start_time)))

            if debug_mode:
                print("----- validating model ...")
                for idx in range(0, num_val, batch_size):
                    latents = sess.run(latent_val)
                    save_images_from_batch(latents, result_imgs_dir, idx)

            if (epoch+1) % save_epochs == 0:
                print("----- saving model  ...")
                saver.save(sess, os.path.join(checkpt_dir, "model.cpkt"), global_step=global_step)
                save_list(os.path.join(result_loss_dir, "total_loss"), total_loss_list)
                save_list(os.path.join(result_loss_dir, "grads_loss"), grad_loss_list)
                save_list(os.path.join(result_loss_dir, "vggs_loss"), vgg_loss_list)
                save_list(os.path.join(result_loss_dir, "order_loss"), order_loss_list)

        # -------------------------------- stage two --------------------------------
        for epoch in range(0, n_epochs2):
            start_time = time.time()
            epoch_loss, n_iters = 0, 0
            avg_grads, avg_vggs, avg_orders, avg_quanti = 0, 0, 0, 0
            for step in range(0, num, batch_size):
                _, loss, grads, vggs, orders, quants = sess.run([train_op2, loss_op2, grads_loss, vgg_loss, global_order_loss, quantization_loss])
                epoch_loss += loss
                avg_grads += grads
                avg_vggs += vggs
                avg_orders += orders
                avg_quanti += quants
                n_iters += 1
                # iteration information
                if n_iters % display_steps == 0:
                    tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print("%s >> [%d/%d] iter: %d  loss: %4.4f" % (tm, epoch+n_epochs1, n_epochs1+n_epochs2, n_iters, loss))

            # epoch information
            epoch_loss = epoch_loss / n_iters
            avg_grads = avg_grads / n_iters
            avg_vggs = avg_vggs / n_iters
            avg_orders = avg_orders / n_iters
            avg_quanti = avg_quanti / n_iters
            total_loss_list.append(epoch_loss)
            grad_loss_list.append(avg_grads)
            vgg_loss_list.append(avg_vggs)
            order_loss_list.append(avg_orders)
            quanti_loss_list.append(avg_quanti)
            print("[*] ----- Epoch: %d/%d | Loss: %4.4f | Time-consumed: %4.3f -----" %
                  (epoch+n_epochs1, n_epochs1+n_epochs2, epoch_loss, (time.time() - start_time)))

            if reporter != None: # reporter for hyperparater tuning
                    reporter(total_loss=epoch_loss)

            if debug_mode:
                print("----- validating model ...")
                for idx in range(0, num_val, batch_size):
                    latents = sess.run(latent_val)
                    save_images_from_batch(latents, result_imgs_dir, idx)

            if (epoch+1) % save_epochs == 0:
                print("----- saving model  ...")
                saver.save(sess, os.path.join(checkpt_dir, "model.cpkt"), global_step=global_step)
                save_list(os.path.join(result_loss_dir, "total_loss"), total_loss_list)
                save_list(os.path.join(result_loss_dir, "grads_loss"), grad_loss_list)
                save_list(os.path.join(result_loss_dir, "vggs_loss"), vgg_loss_list)
                save_list(os.path.join(result_loss_dir, "order_loss"), order_loss_list)
                save_list(os.path.join(result_loss_dir, "quant_loss"), quanti_loss_list)



        # stop data queue
        coord.request_stop()
        coord.join(threads)
        total_training_time = time.time() - init_start_time
        print("Training finished! consumes %f sec" % (total_training_time))

        write_result_file(save_path=os.path.join(RESULT_OUTPUT_DIR, "training.txt"),
                                total_time=total_training_time,
                                train_data_dir=DIR_TO_TRAIN_SET,
                                valid_data_dir=DIR_TO_VALID_SET,
                                train_num=train_num,
                                valid_num=valid_num,
                                num_parameters=num_parameters,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                epoch_num1=n_epochs1,
                                epoch_num2=n_epochs2
                                )


    return None


def evaluate(test_list, checkpoint_dir):
    print('Running ColorEncoder -Evaluation!')
    record_dir = os.path.join(RESULT_OUTPUT_DIR, 'evaluate_tfrecords')
    save_dir_test_gray = os.path.join(RESULT_OUTPUT_DIR, "test_result_imgs", "invertible_gray")
    save_dir_test_color = os.path.join(RESULT_OUTPUT_DIR, "test_result_imgs", "restored_rgb")
    exists_or_mkdir(RESULT_GRAPH_DIR, need_remove=True)
    exists_or_mkdir(RESULT_OUTPUT_DIR)
    exists_or_mkdir(record_dir)
    exists_or_mkdir(save_dir_test_color)
    exists_or_mkdir(save_dir_test_gray)

    # --------------------------------- set model ---------------------------------
    # data fetched within range: [-1,1]
    if RUN_Encoder:
	    # input_imgs, target_imgs, num = input_producer(test_list, 3, batch_size, need_shuffle=False)
        input_imgs, target_imgs, num = tfrecord_input_producer(test_list,
                                                            record_dir,
                                                            3,
                                                            IMG_SHAPE[0],
                                                            batch_size,
                                                            need_shuffle=False)
        latent_imgs = encode(input_imgs, 1, is_train=False, reuse=False)
    else:
	    # input_imgs, target_imgs, num = input_producer(test_list, 1, batch_size, need_shuffle=False)
        input_imgs, target_imgs, num = tfrecord_input_producer(test_list,
                                                            record_dir,
                                                            1,
                                                            IMG_SHAPE[0],
                                                            batch_size,
                                                            need_shuffle=False)
        restored_imgs = decode(input_imgs, out_channels, is_train=False, reuse=False)

    # --------------------------------- evaluation ---------------------------------
    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    test_num = NUMBER_OF_SAMPLES if SAMPLE_TEST_MODE else len(test_list)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model weights from previously saved model
        check_pt = tf.train.get_checkpoint_state(checkpoint_dir)
        if check_pt and check_pt.model_checkpoint_path:
            saver.restore(sess, check_pt.model_checkpoint_path)
            print('model is loaded successfully.')
        else:
            print('# error: loading checkpoint failed.')
            return None

        # write summary
        summary_writer = tf.summary.FileWriter(RESULT_GRAPH_DIR, graph=sess.graph)

        start_time = time.time()
        print("Total images: %d" % test_num)
        print("Image Shape: {}".format(IMG_SHAPE))
        print("Encoder is running... Converting RGB --> grayscale..." if RUN_Encoder == True else "Decoder is running... Converting grayscale --> RGB...")

        cnt = 0
        while not coord.should_stop() and cnt <= test_num:
            tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            print('%s evaluating: [%d - %d]' % (tm, cnt, cnt+batch_size))
            if RUN_Encoder:			# save the synthesized invertible grayscale
                invertible_gray_imgs = sess.run(latent_imgs)
                save_images_from_batch(invertible_gray_imgs, save_dir_test_gray, cnt)
            else:							# save the restored color images
                color_imgs = sess.run(restored_imgs)
                save_images_from_batch(color_imgs, save_dir_test_color, cnt)
            cnt += batch_size
            if cnt >= test_num:
                coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        total_testing_time = time.time() - start_time
        print("Testing finished! consumes %f sec" % (total_testing_time))

        write_result_file(save_path=os.path.join(RESULT_OUTPUT_DIR, "testing.txt"),
                                total_time=total_testing_time,
                                test_data_dir=DIR_TO_TEST_SET,
                                test_num=test_num,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                epoch_num1=n_epochs1,
                                epoch_num2=n_epochs2
                                )

if __name__ == "__main__":
    import argparse
    from config import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    args = parser.parse_args()
    TRAINING_MODE = False if args.mode != 'train' else True
    TEST_ALL = True if args.mode == 'test-all' else False

    if args.mode == 'train':
        train_list = gen_list(DIR_TO_TRAIN_SET)
        val_list = gen_list(DIR_TO_VALID_SET)
        print("Loading train images from {}".format(DIR_TO_TRAIN_SET))
        print("Loading validation images from {}".format(DIR_TO_VALID_SET))
        train(train_list, val_list)
    elif args.mode == 'test':
        test_list = gen_list(DIR_TO_TEST_SET)
        print("Loading test images from {}".format(DIR_TO_TEST_SET))
        evaluate(test_list, EVAL_CHECKPOINT_DIR)
    elif args.mode == 'tune':
        print("Finetunning parameters....")
        ray.init()
        train_list = gen_list(DIR_TO_TRAIN_SET)
        val_list = gen_list(DIR_TO_VALID_SET)
        tune.run(train(train_list, val_list),
            name='l2 lamda',
            stop={"total_loss": 0},
            config={
                "lamda": tune.grid_search([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
                })
    elif args.mode == 'test-all':
        # Run encoder
        print('Running Encoder... {}'.format(RUN_Encoder))
        test_list = gen_list(DIR_TO_TEST_SET)
        print("Loading test images from {}".format(DIR_TO_TEST_SET))
        evaluate(test_list, EVAL_CHECKPOINT_DIR)
        # Run decoder
        DIR_TO_TEST_SET=os.path.join(RESULT_STORAGE_DIR, 'output/test_result_imgs/invertible_gray')
        print('Running Encoder... {}'.format(RUN_Encoder))
        test_list = gen_list(DIR_TO_TEST_SET)
        print("Loading test images from {}".format(DIR_TO_TEST_SET))
        evaluate(test_list, EVAL_CHECKPOINT_DIR)
    else:
        raise Exception("Unknow --mode")
