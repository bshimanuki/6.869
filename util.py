import tensorflow as tf

from constants import DATA_PREFIX, NUM_CHANNELS, TYPE, IMAGE_SIZE


def get_categories():
    # Get list of categories, and mapping from category to index
    with open('development_kit/data/categories.txt') as f:
        categories = []
        category_to_index = {}
        for line in f:
            name, cat = line.split()
            cat = int(cat)
            categories.append(name[3:]) # skip '/./'
            category_to_index[name] = cat
    return (categories, category_to_index)


def get_files(partition, all_categories, target_categories=[], n=None):
    """

    :param partition: String matching folder containing images. Valid values are 'test', 'train' and 'val'
    :param all_categories: All categories that images can be classified in.
    :param target_categories: Target categories. Defaults to all categories if not specified.
    :param n: Target number of examples. Defaults to infinity if not specified.
    :return:
    """
    target_categories = set(target_categories)
    files = []
    labels = []
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
            if not target_categories or all_categories[cat] in target_categories:
                if n is None or num <= n:
                    files.append(DATA_PREFIX + name)
                    labels.append(cat)
    queue = tf.train.slice_input_producer([files, labels], shuffle=True)
    images = tf.read_file(queue[0])
    labels = queue[1]
    images = tf.image.decode_jpeg(images, channels=NUM_CHANNELS)
    images = tf.cast(images, TYPE)
    images.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    return images, labels, files