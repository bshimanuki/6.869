import pickle
import numpy as np

def conf(prediction_file, labels_file):
    with open(prediction_file, 'rb') as pred_file, open(labels_file,'rb') as label_file:
        pred = pickle.load(pred_file)
        labels = pickle.load(label_file)
        l = len(pred)
        mat = np.zeros((100,100))
        for i in range(l):
            lab = np.zeros((100,1))
            lab[labels[i]] = 1
            avg = np.average(pred[i], axis=0)
            avg = np.reshape(avg, (1, 100))
            mat += np.matmul(lab,avg)
        with open('confusion_matrix', 'wb') as f:
            np.savetxt(f, mat, delimiter= ',')

            
