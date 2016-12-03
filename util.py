import tensorflow as tf
import glob

from constants import ALL_CATEGORIES, DATA_DIR, NUM_CHANNELS, TYPE, IMAGE_RESIZED_SIZE, IMAGE_CROPPED_SIZE, IMAGE_MEAN, SEED, \
    FLAG_RESIZE_AND_CROP, FLAG_DEMEAN, FLAG_NORMALIZE, FLAG_RANDOM_FLIP_LR, IMAGE_IMPORT_SIZE, CHECKPOINT_DIRECTORY, \
    BATCH_SIZE, IMAGE_FINAL_SIZE


def get_input(partition, target_categories=[], n=None):
    """

    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param target_categories: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    files, labels = get_files_and_labels(partition, target_categories, n)
    queue = tf.train.slice_input_producer([files, labels], shuffle=True)
    image = tf.read_file(queue[0])
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
    return image, label

def get_files_and_labels(partition, target_categories=[], n=None):
    target_categories = set(target_categories)
    files = []
    labels = []

    # Retrieve test data
    if partition == 'test':
        files = glob.glob(DATA_DIR + ('%s/*.jpg' % partition))
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


# TODO: get rid of this
def save(checkpoint_file):
    with tf.Session() as session:
        x = tf.Variable([42.0, 42.1, 42.3], name='x')
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='y')
        session.run(tf.initialize_all_variables())

        print(session.run(tf.all_variables()))
        saver = tf.train.Saver([x, y])
        saver.save(session, checkpoint_file)

# TODO: get rid of this
def restore(checkpoint_file):
    # Basic Example:
    # x = tf.Variable(-1.0, validate_shape=False, name='x')
    # y = tf.Variable(-1.0, validate_shape=False, name='y')

    # x = tf.Variable(TYPE, shape=(BATCH_SIZE, IMAGE_FINAL_SIZE, IMAGE_FINAL_SIZE, NUM_CHANNELS), name='input')
    y = tf.Variable(0.0, name='labels')
    z = tf.Variable([0,0,0], name='z')
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file)
        print(session.run(tf.all_variables()))

# TODO: get rid of this
def read_checkpoint_file(file_name):
    reader = tf.train.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))

def make_submission_file(checkpoint_path, partition='test', config=tf.ConfigProto(device_count={'GPU':0})):
    data_files = glob.glob('data/images/%s/*.jpg' % partition)
    data_files.sort()
    output_name = 'submissions/' + checkpoint_path + '.txt'

    # Variables to restore
    # for each image, its % confidence prediction for each category

    with tf.Session(config=config) as sess:
        # saver = tf.train.Saver()
        # saver.restore(sess=sess, save_path=checkpoint_path)
        # print("Restored variables from checkpoint %s" % checkpoint_path)

        values, indices = tf.nn.top_k(predictions, k=5, sorted=True, name=name)

        with open (output_name, 'w') as output_file:
            for data_file in data_files:
                short_data_file = '/'.join(data_file.split('/')[-2:])
                output_line = [short_data_file,]
                output_file.write(' '.join(map(str,output_line)) + '\n')
        output_file.close()

# make_submission_file(checkpoint_path)
# checkpoint_file = CHECKPOINT_DIRECTORY + 'test2'
checkpoint_file = CHECKPOINT_DIRECTORY + 'checkpoint__135b22cff320da6188ef72cb6e7f1ee743fa2e15__2016-11-26_20:06:01-200'
read_checkpoint_file(checkpoint_file)
# # save(checkpoint_file=checkpoint_file)
# restore(checkpoint_file=checkpoint_file)

# NOTES
# - need to give path AND file name, for both save and restore
# - only restores the variables that you initialize
# - NEXT: try to restore a variable in saved checkpoint files