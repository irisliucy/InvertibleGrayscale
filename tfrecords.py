import os
from PIL import Image
import tensorflow as tf
from util import *
tf.logging.set_verbosity(tf.logging.ERROR) # suppress annoying tf warnings

class SaveRecord(object):
    """ Save images to tfrecord files
    Args:
        fileDir (string): images' path
        recordDir (string): path of the TFRecord files
        imageSize (int): image size, assuming the x, y of an image is the same
    """
    def __init__(self, recordDir, fileDir, imageSize, batchSize):
        self._imageSize = imageSize
        self._batchSize = batchSize

        trainRecord = os.path.join(recordDir,'train.tfrecord')
        validRecord = os.path.join(recordDir,'valid.tfrecord')

        fileNum = len(fileDir)
        print('the count of images is ' + str(fileNum))

        trainImages = [s.split(' ')[0] for s in fileDir]

#       save data to destinated path
        self.save_data_to_record( fileDir = fileDir, datas = trainImages, recordname = trainRecord)
        print('A total of {} image(s) have been writted to TFRecord files!'.format(str(fileNum)))

    def save_data_to_record(self, fileDir, datas, recordname):
        writer = tf.python_io.TFRecordWriter(recordname) # open the TFRecords file

        for var in datas:
            filename = var.split(' ')[0]
            label = var.split(' ')[-1]
            print("imglist ==> {} \nlabelist ==> {}".format(filename, label))
            image = Image.open(filename)                # open the image
            image = image.resize((self._imageSize,self._imageSize))
            imageArray = image.tobytes()               # convert to bytes

            label = Image.open(label)
            label = label.resize((self._imageSize,self._imageSize))
            labelArray = label.tobytes()

            # Create an example protocol buffer
            example = tf.train.Example(features = tf.train.Features(feature = {
                      'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [imageArray]))
                      ,'label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [labelArray]))}))
            writer.write(example.SerializeToString()) # Serialize to string and write on the file
        writer.close()

    @staticmethod
    def read_and_decode(self, filename, channels):
        """ Read the TFRecords file
        Args:
            filename (string): TFRecord filepath
        """
        print('\nReading and decoding....')
    #    create a queue to hold the filenames
        tfrecord_filename = os.path.join(filename,'train.tfrecord')
        filename_queue = tf.train.string_input_producer([tfrecord_filename])

        reader = tf.TFRecordReader() # define a reader
        _, serialized = reader.read(filename_queue)   # return filename and file

    #    define a decoder
        features = tf.parse_single_example(serialized = serialized, features = {
            'image' : tf.FixedLenFeature([], tf.string),
            'label' : tf.FixedLenFeature([], tf.string)})

        image = tf.decode_raw(features['image'], tf.uint8) # convert the data from string to numbers
        label = tf.decode_raw(features['label'], tf.uint8)

    #    preprocessing
        def preprocessing(input):
            proc = tf.cast(input, tf.float32)
            proc = tf.reshape(proc, [self._imageSize, self._imageSize, channels])
            # normalization
            proc = proc / 127.5 - 1
            return proc

        # output pixel's range : [-1, 1]
        image = preprocessing(image)
        label = preprocessing(label)

    #    batching
        img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size=self._batchSize,
                                                    num_threads=1,
                                                    capacity=64,
                                                    min_after_dequeue=60)

        print('\n tfrecord is read and decoded...')
        return img_batch, label_batch
