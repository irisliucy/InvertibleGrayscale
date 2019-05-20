import os
from PIL import Image
import tensorflow as tf
from util import *

class SaveRecord(object):
    """ Save images to tfrecord files
    Args:
        fileDir (string): path of images
        recordDir (string): path of the TFRecord files
        imageSize (int): image size, assuming the x, y of an image is the same
    """
    def __init__(self, recordDir, fileDir, imageSize):
        self._imageSize = imageSize

        trainRecord = os.path.join(recordDir,'train.tfrecord')
        validRecord = os.path.join(recordDir,'valid.tfrecord')

#        obtain the file list
        filenames = os.listdir(fileDir)
        np.random.shuffle(filenames)
        fileNum = len(filenames)
        print('the count of images is ' + str(fileNum))

#        obtain the split for train to test, 4:1
        # splitNum = int(fileNum * 0.8)
        # trainImages = filenames[ : splitNum]
        # validImages = filenames[splitNum : ]

#       save data to destinated path
        self.save_data_to_record( fileDir = fileDir, datas = trainImages, recordname = trainRecord)
        self.save_data_to_record(fileDir = fileDir,datas = validImages, recordname = validRecord)

    def save_data_to_record(self, fileDir, datas, recordname):
        writer = tf.python_io.TFRecordWriter(recordname) # open the TFRecords file

        for var in datas:
            filename = os.path.join(fileDir, var)
            label = int(os.path.basename(var).split('_')[0])
            image = Image.open(filename)                # open the image
            image = image.resize((self._imageSize,self._imageSize))
            imageArray = image.tobytes()               # convert to bytes

            # Create an example protocol buffer
            example = tf.train.Example(features = tf.train.Features(feature = {
                      'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [imageArray]))
                      ,'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))}))
            writer.write(example.SerializeToString()) # Serialize to string and write on the file
            print('Image has been writted to TFRecord file!')
        writer.close()

    # for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    #     example = tf.train.Example()
    #     example.ParseFromString(serialized_example)
    #
    #     image = example.features.feature['image'].bytes_list.value
    #     # label = example.features.feature['label'].int64_list.value
    #     # TODO: Add some preprocessing steps
    #     print image, label

    def read_and_decode(filename):
        """ Read the TFRecords file
        Args:
            filename (string): TFRecord filepath
        """
    #    create a queue to hold the filenames
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader() # define a reader
        _, serialized = reader.read(filename_queue)   # return filename and file

    #    define a decoder
        features = tf.parse_single_example(serialized = serialized, features = {
            'image' : tf.FixedLenFeature([], tf.string),
            'label' : tf.FixedLenFeature([], tf.string)})

        image = tf.decode_raw(features['image'], tf.uint8) # convert the data from string to numbers
        image= tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3]) # reshape data to original shape
    #    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #    image = tf.cast(features['image'], tf.string)
        label = tf.cast(features['label'], tf.int32)

    #    batching
        img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size=BATCH_SIZE, capacity=2000,
                                                    min_after_dequeue=1000)

        return img_batch, label_batch
