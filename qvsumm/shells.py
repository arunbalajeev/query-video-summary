'''
 Implementation of the objectives used in
 Arun Balajee Vasudevan*, Michael Gygli*, Anna Volokitin, Luc Van Gool - Query-adaptive Video Summarization via Quality-aware Relevance Estimation. ACM Multimedia 2017
'''
__author__ = "Arun Balajee Vasudevan"
__email__ = "arunv@vision.ee.ethz.ch"

import numpy as np
import gm_submodular
import gm_submodular.example_objectives as ex
import model
import theano
import lasagne
import scipy.spatial.distance as dist
import cv2
from qvsumm import config

class Summ(gm_submodular.DataElement):
    '''
    Defines a class Summ.
    For inference, this needs the function get_querylen(), getDistances(), getCosts(), vggmodel(), load_dataset() and get_mfeatures().
    '''
    budget = 5

    def __init__(self, query, imagenames, rel_scores, int_scores):
        self.query = query
        self.imagenames = imagenames
        self.int_scores = int_scores
        self.rel_scores = rel_scores
        self.querylen = self.get_querylen()
        self.Y = self.get_Y()
        self.dist_v = self.get_mfeatures()

    def get_querylen(self):
        valid_array = self.rel_scores
        return len(valid_array)

    def get_Y(self):
        return np.array(range(self.querylen))

    def getCosts(self):
        return np.ones((self.querylen))

    def getDistances(self):
        d = dist.squareform(self.dist_v)
        return np.multiply(d, d)

    def vggmodel(self):
        '''
        Load the VGG19 with pretrained weights
        :return: fc7 layer
        '''
        network = model.build_vggmodel(batch_size=1)
        model.set_vggweights(network['fc7'], config.get('paths', 'vgg_weight_file'), 36)
        prediction = lasagne.layers.get_output(network['fc7'])
        val_fn = theano.function([network['input_layer'].input_var], [prediction])
        return val_fn

    def load_dataset(self, frames):
        '''
        Preprocess the set of images
        '''
        i = 0
        # frames=self.imagenames
        image_size = 224
        data = np.zeros((len(frames), 3, image_size, image_size), dtype=np.float32)
        MEAN_PIXEL = [103.939, 116.779, 123.68]
        p = 0
        for i in range(len(frames)):
            image = frames[i]
            im = cv2.imread(image)
            # im = im[:,:,::-1]
            im = cv2.resize(im, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            im = im - MEAN_PIXEL
            data[i, :, :, :] = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
        X_train = data
        return X_train

    def get_mfeatures(self):
        '''
        Compute Spatial distance between features in fc7 space
        '''
        frames = self.imagenames
        score_fn = self.vggmodel()
        m_features = np.zeros((len(frames), 4096), dtype=np.float32)
        for m, p in enumerate(frames):
            path = []
            path.append(p)
            X = self.load_dataset(path)
            err = score_fn(X)
            m_features[m, :] = err[0]
        return dist.pdist(m_features)


def quality_shell(S):
    '''
    Quality scoring shell Eq.
    :param S: Summ with interestingness scores
    :return: quality objective
    '''
    valid_array = S.int_scores
    mn = min(valid_array);
    stdv = np.std(valid_array);
    a = np.array([(item - mn) / stdv for item in valid_array])
    return (lambda X: (np.sum(a[i] for i in X)))


def similarity_shell(S):
    '''
    Query similarity shell Eq.
    :param S: Summ with relevance scores
    :return: similarity objective
    '''
    valid_array = S.rel_scores
    mn = min(valid_array);
    stdv = np.std(valid_array);
    a = np.array([(item - mn) / stdv for item in valid_array])
    return (lambda X: (np.sum(a[i] for i in X)))


def diversity_shell(S):
    '''
    Diversity shell Eq.
    :param S: Summ DataElement
    :return: diversity objective
    '''
    frames = S.imagenames
    score_fn = S.vggmodel()
    features = np.zeros((len(frames), 4096), dtype=np.float32)
    for m, p in enumerate(frames):
        path = []
        path.append(p)
        X = S.load_dataset(path)
        err = score_fn(X)
        features[m, :] = err[0]

    def square(list):
        return [i ** 2 for i in list]

    floatvec = lambda x: np.array([float(i) for i in x])
    dist = lambda x, y: np.sqrt(
        np.sum(square(floatvec(x) / float(np.linalg.norm(x)) - floatvec(y) / float(np.linalg.norm(y)))))
    c = lambda x, y: dist(features[x, :], features[y, :])
    b = lambda i, X: 5 if i == 0 else min([c(X[i], X[j]) + 1e-4 for j in range(i)])
    return (lambda X: (np.sum([b(i, X) for i in range(len(X))])))
