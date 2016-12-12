import numpy as np
import random

def gradcheck_naive(f,x):
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx,grad = f(x)
    h = 1e-4

    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] += h
        random.setstate(rndstate)
        fxh, _ = f(x)
        x[ix] -= 2*h
        random.setstate(rndstate)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh)/2/h

        reldiff = abs(numgrad-grad[ix])/max(1,abs(numgrad),abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" %(grad[ix],numgrad))
            return
        it.iternext()
    print("Gradient check passed")

def sanity_check():
    quad = lambda x : (np.sum(x ** 2),x*2)
    print("Running sanity checks")
    gradcheck_naive(quad,np.array(123.456))
    gradcheck_naive(quad,np.random.randn(3,))
    gradcheck_naive(quad,np.random.randn(4,5))
            
if __name__ == "__main__":
    sanity_check()
        
