import autograd.numpy as np
import math

def rastrigin(x,y):
    return 20 + (x**2 - 10*math.cos(2*(math.pi)*x)) + (y**2 - 10*math.pi(2*(math.pi)*y))

def sphere(x,y):
    return x**2 + y**2

def rosen(x,y):
    return 100.0*(y-x**2.0)**2.0 + (1-x)**2.0

def beale(x,y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def holder(x,y):
    return -abs(np.sin(x)*np.cos(y)*np.exp(abs(1 - (np.sqrt(x**2 + y**2)/np.pi))))

def camel(x,y):
    return 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2
