import numpy as np
import random

from cs224.data_utils import *
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q3_sgd import load_saved_params

def getSentenceFeature(tokens,wordVectors,sentence):
    sentVector = np.zeros((wordVectors.shape[1],))
    indices = [tokens[word] for word in sentence]
    sentVector = np.mean(wordVectors[indices,:],axis=0)

    return sentVector

def softmaxRegression(features, labels, weights, regularization = 0.0, nopredictions = False):
    prob = softmax(features.dot(weights))
    if len(features.shape)>1:
        N = features.shape[0]
    else:
        N = 1
    cost = np.sum(-np.log(prob[range(N),labels]))/N
    cost += 0.5 * regularization * np.sum(weights ** 2)

    grad = np.array(prob)
    grad[range(N),labels] -= 1.0
    grad = features.T.dot(grad)/N
    grad += regularization*weights
    if N > 1:
        pred = np.argmax(prob,axis=1)
    else:
        pred = np.argmax(prob)
    if nopredictions:
        return cost,grad
    else:
        return cost,grad,pred
def accuracy(y,yhat):
    assert(y.shape == yhat.shape)
    return np.sum(y== yhat)*100.0 / y.size

def softmax_wrapper(features,labels,weights,regularization = 0.0):
    cost, grad, _ = softmaxRegression(features,labels,weights,regularization)
    return cost,grad

def sanity_check():
    random.seed(312159)
    np.random.seed(265)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    _, wordVectors0, _ = load_saved_params()
    wordVectors = (wordVectors0[:nWords,:]+ wordVectors0[nWords:,:])
    dimVectors = wordVectors.shape[1]

    dummy_weights = 0.1* np.random.randn(dimVectors,5)
    dummy_features = np.zeros((10,dimVectors))
    dummy_labels = np.zeros((10,),dtype = np.int32)
    for i in range(10):
        words, dummy_labels[i] = dataset.getRandomTrainSentence()
        dummy_features[i,:] = getSentenceFeature(tokens,wordVectors,words)
    print("Gradient check for softmax regression")
    gradcheck_naive(lambda weights : softmaxRegression(dummy_features,dummy_labels,weights,1.0,nopredictions = True),dummy_weights)
    print("\n===Results===")
    print(softmaxRegression(dummy_features,dummy_labels,dummy_weights,1.0))

if __name__ == "__main__":
    sanity_check()
