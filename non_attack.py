import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg

from non_preprocess import preprocess_for_model
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, inception_v1, inception
from PIL import Image
import glob
from scipy.misc import imresize
from progressbar import *

CHECKPOINTS_DIR = './checkpoints/'
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


def load_images_with_true_label(input_dir, batch_shape):
    # images = []
    images = np.zeros(batch_shape)
    filenames = []
    true_labels = []
    idx = 0
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['trueLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        # image = imread(os.path.join(input_dir, filename), mode='RGB')
        # images.append(image)
        image = Image.open(os.path.join(input_dir,filename))
        image = image.resize((224, 224), Image.BILINEAR)
        images[idx,:,:,:] = image
        filenames.append(filename)
        true_labels.append(filename2label[filename])
        idx += 1
        if idx == batch_shape[0]:
            images = np.array(images)
            yield filenames, images, true_labels
            filenames = []
            # images = []
            images = np.zeros(batch_shape)
            true_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, true_labels




def load_images_no_label(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    labels = []
    batch_size = batch_shape[0]
    idx = 0
    filepaths = []
   # widgets = ['save_atk_image:', Percentage(), ' ',Bar('*'),' ',Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
   # pbar = ProgressBar(widgets=widgets)
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        # image = imread(filepath, mode='RGB').astype(np.float) / 255.0
        filepaths.append(filepath)
        idx +=1
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '.*jpg')):
        filepaths.append(filepath)
        idx +=1
    im = 0
    for i in range(idx):
        image = Image.open(filepaths[i])
        image = image.resize((224,224), Image.BILINEAR)
        # image = image()
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[i%batch_size, :, :, :] = (np.array(image).astype(np.float) / 255.0) * 2.0 - 1.0
        # images.append(image)
        filename = os.path.basename(filepaths[i])
        filenames.append(filename)
        label = filepath.split('/')[-2]
        labels.append(label)
        im += 1
        if im == batch_size:
            yield filenames, images, labels
            images = np.zeros(batch_shape)
            filenames = []
            labels=[]
            im = 0
    if im > 0:
        images=np.array(images)
        yield filenames, images, labels


def save_ijcai_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i].clip(0,255)).astype(np.uint8)
        # resize back to [299, 299]
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')
        # print('{} image saved'.format(os.path.join(output_dir, filename)))


def non_target_graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    alpha = eps / FLAGS.num_iter
    num_classes = 110

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            x, num_classes=num_classes, is_training=False, scope='InceptionV1')

    image = (((x + 1.0) * 0.5) * 255.0)
    processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
            processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, scope='resnet_v1_50')

    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

    # image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
    processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

    pred = tf.argmax(end_points_inc_v1['Predictions'] + end_points_vgg_16['probs'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)

    logits = (end_points_inc_v1['Logits']  + end_points_vgg_16['logits']) / 2.0
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
    noise = FLAGS.momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad, eps_inside):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def non_stop(x, y, i, x_max, x_min, grad):
  return tf.less(i, FLAGS.num_iter)

# attack function for ijcai
def non_attack(input_dir, output_dir):
    eps = FLAGS.max_epsilon*2.0/255.0
    print('output_dir is%',FLAGS.output_dir)

    gpus = np.array(FLAGS.gpu.split(',')).astype('int')

    batch_shape = [FLAGS.batch_size, 224, 224, 3]
    n_gpus = len(gpus)
    # processed_img =
    all_imgs = glob.glob(os.path.join(FLAGS.input_dir, '/*/*.jpg'))

    with tf.Graph().as_default():
        raw_inputs = tf.placeholder(tf.uint8, shape=[None, 224, 224, 3])

        # flists = set([f for f in os.listdir(FLAGS.input_dir) if 'jpg' in f])
        # if FLAGS.use_existing == 1:
        #     flists_existing = set(
        #         [os.path.split(f)[-1] for f in glob.glob(os.path.join(FLAGS.output_dir, '/*/*.jpg')) if 'jpg' in f])
        #     newfiles = list(flists.difference(flists_existing))
        #     newfiles = [os.path.join(FLAGS.input_dir, f) for f in newfiles]
        # else:
            # newfiles = [os.path.join(FLAGS.inpuzt_dir,f) for f in flists]
        newfiles = glob.glob(os.path.join(FLAGS.input_dir, '*/*.jpg'))
        print('creating %s new files' % (len(newfiles)))
        if len(newfiles) == 0:
            return

        # set model type for training
        model_type = 'InceptionV1'
        processed_imgs = preprocess_for_model(raw_inputs, model_type=model_type)
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        # y = tf.constant(np.zeros([bs_single]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(non_stop, non_target_graph, [x_input, y, i, x_max, x_min, grad])
        # s0 = tf.train.Saver(slim.get_model_variables(scope='InceptionResV1'))
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])
            # if FLAGS.target
            for filenames, raw_images, true_labels in load_images_no_label(input_dir, batch_shape):
                processed_imgs_ = sess.run(processed_imgs, feed_dict={raw_inputs: raw_images})
                adv_images = sess.run(x_adv, feed_dict={x_input: processed_imgs_})
                save_ijcai_images(adv_images, filenames, output_dir)


if __name__ == '__main__':
    # input_dir = '/home/zhuxudong/competition/ijcai2019/dev_data/'
    # output_dir = '/home/zhuxudong/competition/ijcai2019Ω/output/'
    non_attack(FLAGS.input_dir, FLAGS.output_dir)
    pass
