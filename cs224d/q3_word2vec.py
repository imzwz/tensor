import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid,sigmoid_grad
from q2_gradcheck import gradcheck_naive

def normalizeRows(x):
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1,2]]))
    print(x)
    assert (np.amax(np.fabs(x - np.array([[0.6,0.8],[0.4472136,0.89442719]]))) <= 1e-6)
    print("")

def softmaxCostAndGradient(predicted,target,outputVectors,dataset):
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])
    delta = probabilities
    delta[target] -= 1
    N = delta.shape[0]
    D = predicted.shape[0]
    grad = delta.reshape((N,1))*predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()
    return cost,gradPred,grad

def negSamplingCostAndGradient(predicted,target,outputVectors,dataset,K=10):
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    indices = [target]
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]
    
    labels = np.array([1]+ [-1 for k in range(K)])
    vecs = outputVectors[indices,:]
    t = sigmoid(vecs.dot(predicted)*labels)
    cost = -np.sum(np.log(t))
    delta = labels * (t-1)
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape((1,predicted.shape[0])))

    for k in range(K+1):
        grad[indices[k]] += gradtemp[k,:]

    return cost,gradPred,grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors,outputVectors,dataset,word2vecCostAndGradient = softmaxCostAndGradient):
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI,:]
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in  contextWords:
        idx = tokens[cwd]
        cc,gp,gg = word2vecCostAndGradient(predicted,idx,outputVectors,dataset)
        cost += cc
        gradOut += gp
        gradIn[currentI,:] += gp

    return cost, gradIn,gradOut

def cbow(currentWord, C, contextWords, tokens,inputVectors,outputVectors,dataset,word2vecCostAndGradient = softmaxCostAndGradient):
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    ###to be continued

    return cost,gradIn,gradOut

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient= softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword,C1,context,tokens,inputVectors,outputVectors,dataset,word2vecCostAndGradient)
        cost += c / batchsize /denom
        grad[:N/2,:] += gin/ batchsize /denom
        grad[N/2:,:] += gout / batchsize / denom
    return cost,grad

def test_word2vec():
    dataset = type('dummy',(),{})()
    def dummySampleTokenIdx():
        return random.randint(0,4)
    def getRandomContext(C):
        tokens = ["a","b","c","d","e"]
        randContext=[]
        for i in range(2*C):
            randContext.append(tokens[random.randint(0,4)])

        return tokens[random.randint(0,4)] , randContext

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0),("b",1),("c",2),("d",3),("e",4)])
    print("Gradient check for skip-gram")

    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram,dummy_tokens,vec,dataset,5),dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram,dummy_tokens,vec,dataset,5,negSamplingCostAndGradient),dummy_vectors)
    print("\nResult====")
    print(skipgram("c",3,["a","b","e","d","b","c"],dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
    




