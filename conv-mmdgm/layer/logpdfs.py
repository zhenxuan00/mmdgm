import numpy as np
import theano
import theano.tensor as T
import math

# library with theano PDF functions

c = - 0.5 * math.log(2*math.pi)

def normal(x, mean, sd):
	return c - T.log(T.abs_(sd)) - (x - mean)**2 / (2 * sd**2)

def normal2(x, mean, logvar):
	return c - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))
    
def standard_normal(x):
	return c - x**2 / 2