import argparse
from alexnet_small import AlexNetSmall
from briannet import BrianNet
from vggnet import VGGNet
from constants import *
from model import Model
import tensorflow as tf
import pickle
from util import *


def run_test(checkpoint_file, model_name):
    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_FINAL_SIZE, IMAGE_FINAL_SIZE, NUM_CHANNELS), name='input')

    # Get test data
    test_data, test_labels = get_input('test', n=IMAGES_PER_CAT)

    print("Got test data successfully")
    test_size = get_size('test')

    keep_prob = 1.
    if model_name == 'AlexNet':
        model = AlexNetSmall(keep_prob)
    elif model_name == 'VGGNet':
        model = VGGNet(keep_prob)
    else:
        raise Exception('Not a valid model name')

    logits, variables = model.model(x)
    prediction = tf.nn.softmax(logits)


    batch_data = tf.train.batch(
        [test_data],
        batch_size=BATCH_SIZE,
        capacity=5*BATCH_SIZE)

    if USE_GPU:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        # Restore variables
        saver.restore(sess=sess, save_path=checkpoint_file)
        print("Restored variables from checkpoint %s" % checkpoint_file)

        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        predictions = []
        n=0
        for step in range(5):

            # Feed dictionary
            _data = sess.run(batch_data)
            test_feed_dict = {x: _data}
            test_predictions = sess.run(prediction, feed_dict=test_feed_dict)
            predictions.extend(test_predictions)
            print('processing batch number: %d' % n)
            n+=1
        filename = checkpoint_file.split('/')
        if filename[-1] == '':
            filename = filename[-2]
        else:
            filename = filename[-1]
        with open('predictions_'+filename, 'wb') as f:
            pickle.dump(predictions, f)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Model Checkpoint File")
    parser.add_argument("-m", "--model", type=str , help="Model Type (AlexNet, VGGNet)")

    args = parser.parse_args()

    run_test(args.file, args.model)
