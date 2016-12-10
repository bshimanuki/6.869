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

    # Get test data
    test_data, test_labels, _ = get_inputs_crop_flip('test')
    print("Retrieved test data successfully")
    sample_size = get_size('test')

    # Initialize model
    keep_prob = 1.
    if model_name == 'AlexNet':
        model = AlexNetSmall(keep_prob)
    elif model_name == 'VGGNet':
        model = VGGNet(keep_prob)
    else:
        raise Exception('Not a valid model name')

    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_FINAL_SIZE, IMAGE_FINAL_SIZE, NUM_CHANNELS), name='input')
    logits, variables = model.model(x)
    prediction = tf.nn.softmax(logits)

    batch_data = tf.train.batch(
        test_data,
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        predictions = []

        for step in range(sample_size // BATCH_SIZE):
            _data = sess.run(batch_data)

            single_prediction = []
            for i in range(8):
                test_feed_dict = {x: _data[i]}
                test_predictions = sess.run(prediction, feed_dict=test_feed_dict)
                single_prediction.append(test_predictions)

            # single_prediction is 8 * BATCH_SIZE * NUM_CAT
            single_prediction = np.array(single_prediction)
            single_prediction = np.swapaxes(single_prediction, 0, 1)

            #single prediction is now BATCH_SIZE * 8 * NUM_CAT

            # average_prediction = np.prod(np.array(single_prediction), axis=0)

#            flat_prediction = np.array(single_prediction).flatten()
#            ind = np.argpartition(flat_prediction, -40)[-40:]
#            indices = ind[np.argsort(flat_prediction[ind])][::-1]
#            indices = np.unique(indices%100)[:5]
#            average_prediction = np.zeros(100)
#            for i in range(5):
#                average_prediction[indices[i]] = 6-i

            predictions.extend(single_prediction.tolist()) 
            # NUM_IMAGES * 8 * NUM_CAT
            print('Processing batch number: %d / %d' % (step, sample_size//BATCH_SIZE))

        # Save predictions in pickle file
        filename = checkpoint_file.split('/')
        if filename[-1] == '':
            prefix = filename[-2]
        else:
            prefix = filename[-1]
        prediction_file = PREDICTIONS_DIR + 'prediction__' + prefix
        with open(prediction_file, 'wb') as f:
            pickle.dump(predictions, f)
        print('Created new pickle file with predictions in %s' % prediction_file)
        
        coord.request_stop()
        coord.join(threads)

    return prediction_file

def run_validation(checkpoint_file, model_name):
    test_data, test_labels, _ = get_inputs_crop_flip('val')
    print("Retrieved validation data successfully")
    sample_size = get_size('val')

    keep_prob = 1.
    if model_name == 'AlexNet':
        model = AlexNetSmall(keep_prob)
    elif model_name == 'VGGNet':
        model = VGGNet(keep_prob)
    else:
        raise Exception('Not a valid model name')

    x = tf.placeholder(TYPE, shape=(BATCH_SIZE, IMAGE_FINAL_SIZE, IMAGE_FINAL_SIZE, NUM_CHANNELS), name='input')
    logits, variables = model.model(x)
    prediction = tf.nn.softmax(logits)

    batch_data, batch_labels = tf.train.batch(
        [test_data, test_labels],
        batch_size=BATCH_SIZE,
        capacity=5*BATCH_SIZE
        )

    if USE_GPU:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        # Restore variables
        saver.restore(sess=sess, save_path=checkpoint_file)
        print("Restored variables from checkpoint %s" % checkpoint_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        predictions = []
        labels = []

        for step in range(sample_size // BATCH_SIZE):
            _data, _labels = sess.run([batch_data, batch_labels])
            _data = np.swapaxes(_data,0,1)
            single_prediction = []
            for i in range(8):
                test_feed_dict = {x: _data[i]}
                test_predictions = sess.run(prediction, feed_dict=test_feed_dict)
                single_prediction.append(test_predictions)

            # single_prediction is 8 * BATCH_SIZE * NUM_CAT
            single_prediction = np.array(single_prediction)
            single_prediction = np.swapaxes(single_prediction, 0, 1)

            #single prediction is now BATCH_SIZE * 8 * NUM_CAT

            # _labels is BATCH_SIZE
            labels.extend(_labels)

            predictions.extend(single_prediction.tolist()) 
            # NUM_IMAGES * 8 * NUM_CAT
            print('Processing batch number: %d / %d' % (step, sample_size//BATCH_SIZE))

        # Save predictions in pickle file
        filename = checkpoint_file.split('/')
        if filename[-1] == '':
            prefix = filename[-2]
        else:
            prefix = filename[-1]
        prediction_file = PREDICTIONS_DIR + 'val_prediction__' + prefix
        labels_file = PREDICTIONS_DIR + 'val_labels__' + prefix
        
        with open(prediction_file, 'wb') as f:
            pickle.dump(predictions, f)
        with open(labels_file, 'wb') as f:
            pickle.dump(labels, f)

        print('Created new pickle file with predictions in %s' % prediction_file)
        
        coord.request_stop()
        coord.join(threads)

    return (prediction_file, labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Model Checkpoint File")
    parser.add_argument("-m", "--model", type=str , help="Model Type (AlexNet, VGGNet)")
    parser.add_argument("-g", "--aggregate", type=str, help="Aggregation method (average, product, max)")
    parser.add_argument("-v", "--validation", action='store_true')
    parser.set_defaults(validation=False)

    args = parser.parse_args()

    # run_test(args.file, args.model)
    if args.validation:
        prediction_file = run_validation(args.file, args.model)
    else:
        prediction_file = run_test(args.file, args.model)
        make_submission_file(prediction_file, args.aggregate)
