import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
import cleverhans

from non_preprocess import preprocess_for_model
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, inception_v1, inception
from PIL import Image
import glob
from scipy.misc import imresize
from progressbar import *
import sys


CHECKPOINTS_DIR = './models/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'use_existing', 0, 'whether reuse existing result')

tf.flags.DEFINE_integer(
    'random_eps', 0, 'whether use random pertubation')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string(
    'gpu','0','')

FLAGS = tf.flags.FLAGS
def tradition(input_dir, output_dir):
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['trueLabel'] for i in range(len(dev))}

    for filename in filename2label:


        # split_line = line.strip().split(",")

        # filename = split_line[0]
        true_label = filename2label[filename]
        # targeted_label = int(split_line[2])

        image_pil = Image.open(os.path.join(input_dir,filename))
        image = np.asarray(image_pil.resize([299, 299], Image.BILINEAR).convert("RGB")).astype(np.float32)


        left = 36
        right = 263
        top = 36
        bottom = 263

        new_image = image.copy()
        n = 1
        for i in range(36):
            new_image[left:right, top:bottom] += image[(left - i):(right - i), top:bottom]
            new_image[left:right, top:bottom] += image[(left + i):(right + i), top:bottom]
            new_image[left:right, top:bottom] += image[left:right, (top - i):(bottom - i)]
            new_image[left:right, top:bottom] += image[left:right, (top + i):(bottom + i)]
            n += 4

        new_image[left:right, top:bottom] /= n
        new_image = new_image.astype(np.uint8)

        Image.fromarray(np.asarray(new_image, np.int8), "RGB").save(os.path.join(output_dir,filename))


if __name__ == '__main__':

    tradition(FLAGS.input_dir, FLAGS.output_dir)
