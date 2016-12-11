import glob
import pickle

import numpy as np
import tensorflow as tf

from constants import *


def get_input(partition, target_categories=[], n=None, shuffle=True):
    """
    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param target_categories: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    files, labels = get_files_and_labels(partition, target_categories, n)
    queue = tf.train.slice_input_producer([files, labels], shuffle=shuffle)
    _file = queue[0]
    image = tf.read_file(_file)
    label = queue[1]
    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
    image = tf.cast(image, TYPE)
    if FLAG_RESIZE_AND_CROP:
        image = tf.image.resize_images(image, [IMAGE_RESIZED_SIZE, IMAGE_RESIZED_SIZE])
        image = tf.image.random_ops.random_crop(image, [IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE, NUM_CHANNELS], seed=SEED)
    else:
        image.set_shape((IMAGE_IMPORT_SIZE, IMAGE_IMPORT_SIZE, NUM_CHANNELS))
    if FLAG_RANDOM_FLIP_LR:
        image = tf.image.random_flip_left_right(image, seed=SEED)
    if FLAG_DEMEAN:
        image = image - IMAGE_MEAN
    if FLAG_NORMALIZE:
        image = image/255.
    if FLAG_ADD_NOISE and partition == 'train':
        image = tf.image.random_hue(image, 0.02)
        image = tf.image.random_saturation(image, 0.9, 1.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_brightness(image, 0.1)
        image += np.random.normal(0, 0.02)
    return image, label, _file

def get_files_and_labels(partition, target_categories=[], n=None):
    target_categories = set(target_categories)
    files = []
    labels = []

    # Retrieve test data
    if partition == 'test':
        files = glob.glob(DATA_DIR + ('%s/*.jpg' % partition))
        labels = [-1] * len(files)
        files.sort()
        return files, labels

    # Retrieve training, validation data and labels
    with open('development_kit/data/%s.txt' % partition) as f:
        for line in f:
            name, cat = line.split()
            num = int(''.join(filter(str.isdigit, name)))
            cat = int(cat)
            """
            Current example is used if:
                (1) no categories were specified, or if its category is in the list of target categories `cats`.
                (2) `n` is not specified, or the number of examples so far is less than the target `n`
            """
            if not target_categories or ALL_CATEGORIES[cat] in target_categories:
                if n is None or num <= n:
                    files.append(DATA_DIR + name)
                    labels.append(cat)
    return files, labels

def get_size(partition, target_categories=[], n=None):
    # TODO: better naming ...
    files, labels = get_files_and_labels(partition, target_categories, n)
    return len(files)


def accuracy(predictions, labels, k=1):
    """Determines the accuracy of the predictions, and prints tally of (top_prediction, label).

    A prediction is considered accurate if the label is among the top k predictions.

    :param predictions: (batch_size, classes) tensor of predictions, with each entry corresponding to the probability
        that an example corresponds to a given class.
    :param labels: batch_size vector of class ids.
    :param k:
    :return: Proportion of accurate predictions, as a 0-dimensional tensor.
    """
    correct = tf.nn.in_top_k(predictions, labels, k)
    # print('\t', Counter(zip(np.argmax(predictions, 1).tolist(), labels)))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def make_submission_file(prediction_file, aggregation_method):
    data_files = glob.glob(DATA_DIR + 'test/*.jpg')
    data_files.sort()

    prefix_list = prediction_file.split('__')[1:]
    prefix = '__'.join(prefix_list)
    output_file = SUBMISSIONS_DIR + 'submission__' + prefix + '__' + aggregation_method+ '.txt'

    with open(prediction_file, 'rb') as pred_f:
        print("Opened prediction file %s" % prediction_file)
        predictions = pickle.load(pred_f)

        with open(output_file, 'w') as out_f:
            print("Created submission file %s" % output_file)
            for i in range(len(predictions)):
                short_data_file = '/'.join(data_files[i].split('/')[-2:])
                output_line = [short_data_file]

                prediction = predictions[i]
                np_prediction = np.array(prediction)
                num_prediction_per_image = len(np_prediction)

                if (aggregation_method == "average"):
                    ensemble_prediction = np.average(np_prediction, axis = 0)
                elif (aggregation_method == "product"):
                    ensemble_prediction = np.prod(np_prediction, axis=0)
                elif (aggregation_method == "max"):
                    ensemble_prediction = np_prediction.max(axis=0)
                else:
                    raise("Invalid aggregation method")
                
                ind = np.argpartition(ensemble_prediction, -5)[-5:]
                indices = ind[np.argsort(ensemble_prediction[ind])][::-1]
                labels = list(map(str, indices))

                output_line.extend(labels)
                out_f.write(' '.join(output_line) + '\n')
        out_f.close()
    pred_f.close()

def get_inputs_crop_flip(partition, target_categories=[], n=None):
    """
    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param target_categories: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    files, labels = get_files_and_labels(partition, target_categories, n)
    queue = tf.train.slice_input_producer([files, labels], shuffle=False)
    _file = queue[0]
    image = tf.read_file(_file)
    label = queue[1]

    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
    image = tf.cast(image, TYPE)

    if FLAG_DEMEAN:
        image = image - IMAGE_MEAN
    if FLAG_NORMALIZE:
        image = image/255.

    image1 = tf.image.crop_to_bounding_box(image, 0, 0, IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE)
    # image1 = tf.image.random_ops.random_crop(image, [IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE, NUM_CHANNELS], seed=SEED)


    image2 = tf.image.crop_to_bounding_box(image, 0, IMAGE_RESIZED_SIZE - IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE)
    image3 = tf.image.crop_to_bounding_box(image, IMAGE_RESIZED_SIZE - IMAGE_CROPPED_SIZE, 0, IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE)
    image4 = tf.image.crop_to_bounding_box(image, IMAGE_RESIZED_SIZE - IMAGE_CROPPED_SIZE, IMAGE_RESIZED_SIZE - IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE, IMAGE_CROPPED_SIZE)

    image5 = tf.image.flip_left_right(image1)
    image6 = tf.image.flip_left_right(image2)
    image7 = tf.image.flip_left_right(image3)
    image8 = tf.image.flip_left_right(image4)


    return [image1, image2, image3, image4, image5, image6, image7, image8], label, _file

def evaluate_predictions(prediction_file, label_file):
    with open(prediction_file, 'rb') as pred_f, open(label_file, 'rb') as label_f:
        print("Opened prediction file %s" % prediction_file)
        predictions = pickle.load(pred_f)
        labels = pickle.load(label_f)

        max_accuracy_1 = 0
        max_accuracy_5 = 0
        average_accuracy_1 = 0
        average_accuracy_5 = 0 
        product_accuracy_1 = 0
        product_accuracy_5 = 0

        for i in range(len(predictions)):
            prediction = predictions[i]
            label = labels[i]

            np_prediction = np.array(prediction)
            num_prediction_per_image = len(np_prediction)

            #average
            avg = np.average(np_prediction, axis = 0)
            ind = np.argpartition(avg, -5)[-5:]
            indices = ind[np.argsort(avg[ind])][::-1]
            if label == indices[0]:
                average_accuracy_1+=1
                average_accuracy_5+=1
            elif label in indices:
                average_accuracy_5+=1

            #product
            prod = np.prod(np_prediction, axis=0)
            ind = np.argpartition(prod, -5)[-5:]
            indices = ind[np.argsort(prod[ind])][::-1]
            if label == indices[0]:
                product_accuracy_1+=1
                product_accuracy_5+=1
            elif label in indices:
                product_accuracy_5+=1

            m = np_prediction.max(axis=0)
            ind = np.argpartition(m, -5)[-5:]
            indices = ind[np.argsort(m[ind])][::-1]
            if label == indices[0]:
                max_accuracy_1+=1
                max_accuracy_5+=1
            elif label in indices:
                max_accuracy_5+=1

        max_accuracy_1/=len(predictions)
        max_accuracy_5/=len(predictions)
        average_accuracy_1/=len(predictions)
        average_accuracy_5/=len(predictions)
        product_accuracy_1/=len(predictions)
        product_accuracy_5/=len(predictions)


        print('max ----- Top 1 : %f, Top5: %f' %(max_accuracy_1, max_accuracy_5))
        print('avg ----- Top 1 : %f, Top5: %f' %(average_accuracy_1, average_accuracy_5))
        print('prod ----- Top 1 : %f, Top5: %f' %(product_accuracy_1, product_accuracy_5))
