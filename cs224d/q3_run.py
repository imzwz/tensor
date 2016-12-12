import random
import numpy as np
from data_utils import *
import matplotlib.pyplot as plt
from q3_word2vec import *
from q3_sgd import *

random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

dimVectors = 10
#Context size
C = 5

random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords,dimVectors)-0.5)/ dimVectors,np.zeros((nWords,dimVectors))),axis=0)
wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram,tokens,vec,dataset,C,negSamplingCostAndGradient), 
        wordVectors,0.3,40000,None,True,PRINT_EVERY=10)
print("sanity check: cost at convergence should be around or below 10")

wordVectors = (wordVectors0[:nWords,:]+ wordVectors0[nWords:,:])
_, wordVectors0, _ = (wordVectors0[:nwords,:]+ wordVectors0[nWords:,:])
visualizeWords = ["the","a","an"]
visuallizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx,:]
temp = (visualizeVecs - np.mean(visualizeVecs,axis =0))
covariance = 1.0/len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])
for i in range(len(visualizeWords)):
    plt.text(coord[i,0],coord[i,1],visualizeWords[i],bbox = dict(facecolor = 'green',alpha=0.1))
plt.xlim((np.min(coord[:,0]),np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]),np.max(coord[:,1])))
plt.savefig('q3_word_vectors.png')
plt.show()
