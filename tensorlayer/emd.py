import numpy as np
import scipy.optimize

#Constraints
def positivity(f):
    return f

def fromSrc(f, wp, i, shape):
    fr = np.reshape(f, shape)
    f_sumColi = np.sum(fr[i,:])
    return wp[i]- f_sumColi

def toTgt(f, wq, j, shape):
    fr = np.reshape(f, shape)
    f_sumRowj = np.sum(fr[:,j])
    return wq[j] - f_sumRowj

def maximiseTotalFlow(f, wp, wq):

    return f.sum() - np.minimum(wp.sum(), wq.sum())

def flow(f, D):
    f = np.reshape(f, D.shape)
    return (f*D).sum()

def groundDistance(x1, x2, norm=2):

    return np.linalg.norm(x1-x2, norm)

def getDistMatrix(s1, s2, norm=2):
    numFeats1 = s1.shape[0]
    numFeats2 = s2.shape[0]
    distMatrix = np.zeros((numFeats1, numFeats2))

    for i in range(0, numFeats1):
        for j in range(0, numFeat2):
            distMatrix[i,j] = groundDistance(s1[i], s2[j], norm)
    return distMatrix

def getDistMatrix(P, Q, D):
    numFeats1 = P[0].shape[0]
    numFeats2 = Q[0].shape[0]
    shape = (numFeats1, numFeats2)
    cons1 = [{'type':'ineq', 'fun': positivity},
            {'type':'eq', 'fun': maximiseTotalFlow, 'args':(P[1],Q[1],)}]
    cons2 = [{'type':'ineq', 'fun': fromSrc, 'args':(P[1], i, shape,)} for i in range(numFeats1)]
    cons3 = [{'type':'ineq', 'fun': toTgt, 'args':(Q[1], j, shape,)} for j in range(numFeats2)]
    cons = cons1 + cons2 + cons3
    F_guess = np.zeros(D.shape)
    F = scipy.optimize.minimize(flow, F_guess, args=(D,), constrains=cons)
    F = np.reshape(F.x, (numFeats1, numFeats2))
    return F

def EMD(F,D):
    return (F*D).sum() / F.sum()

def getEMD(P,Q,norm=2):
    D = getDistMatrix(P[0], Q[0], norm)
    F = getFlowMatrix(P,Q,D)
    return EMD(F,D)

def getExampleSignatures():
    features1 = np.array([[100,40,22],[211,20,2],[32,190,150],[2,100,100]])
    weights1 = np.array([0.4, 0.3, 0.2, 0.1])
    features2 = np.array([[0,0,0], [50,100,80],[255,255,255]])
    weights2 = np.array([0.5, 0.3, 0.2])
    signature1 = (features1, weights1)
    signature2 = (features2, weights2)
    return signature1, signature2

if __name__ == '__main__':
    print("EMD")
    P,Q = getExampleSignatures()
    emd = getEMD(P,Q)
    print("We got:",emd)
    print("C example got 160.54277")
    print("success")





