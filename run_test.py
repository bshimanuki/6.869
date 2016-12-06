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
    test_size = get_size('test')

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
        n = 0
        for step in range(test_size // BATCH_SIZE):
            _data = sess.run(batch_data)
            print(type(_data))
            print(type(_data[0]))
            print(type(_data[0][0]))
            print(type(_data[0][0][0]))

            assert((_data[0].all() == _data[2].all()))
            single_prediction = []
            for i in range(8):
                test_feed_dict = {x: _data[i]}
                test_predictions = sess.run(prediction, feed_dict=test_feed_dict)
                single_prediction.append(test_predictions)

            average_prediction = np.average(np.array(single_prediction), axis=0)
            print(average_prediction)
            print('----')
            print(_data[0])
            assert(average_prediction.tolist()[0] == _data[0])
            predictions.extend(average_prediction.tolist())
            print('Processing batch number: %d' % n)
            n+=1

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Model Checkpoint File")
    parser.add_argument("-m", "--model", type=str , help="Model Type (AlexNet, VGGNet)")

    args = parser.parse_args()

    # run_test(args.file, args.model)
    prediction_file = run_test(args.file, args.model)

    make_submission_file(prediction_file)
