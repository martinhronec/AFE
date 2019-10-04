
"""
interactive_plot.py
##############################################################################
"""


import scipy
import matplotlib.pyplot as plt
import random
import numpy as np

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def draw_random_sample(seed=42):
    random.seed(seed)
    size_sample = 100
    x_sample = scipy.random.uniform(size=size_sample)
    y_sample = scipy.sin(4*x_sample) + scipy.randn(size_sample)/3
    return x_sample, y_sample

xgrid = scipy.linspace(0, 1, 1001)
y_true = scipy.sin(4*xgrid)
x_sample, y_sample = draw_random_sample(42)

def predict(point:float, data_x, data_y, k, polynomial_order=0)-> float:
    """Makes prediction for one point."""
    if polynomial_order == 0:
        return (k @ data_y).sum()/k.sum()
    elif polynomial_order == 1:
        W = np.diag(k)
        B = np.append(np.ones([len(data_x), 1]),data_x.reshape((len(data_x),1)), axis=1)
        b = np.array([1,point])
        return b@np.linalg.inv((B.T@W@B))@(B.T@W@data_y)
    elif polynomial_order == 2:
        W = np.diag(k)
        B = np.append(np.ones([len(data_x), 1]),data_x.reshape((len(data_x),1)), axis=1)
        B = np.append(B,data_x.reshape((len(data_x),1))*data_x.reshape((len(data_x),1)),axis=1)
        b = np.array([1,point,point*point])
        return b@np.linalg.inv((B.T@W@B))@(B.T@W@data_y)

# DEFINITIONS OF KERNELS
# ----------------------------------------------------------------------------    
def knn(k: int, point:float,
        data_x:scipy.ndarray, data_y:scipy.ndarray) -> float:
    idx_sorted = scipy.argsort((data_x-point)*(data_x-point))[:k]
    return data_y[idx_sorted].mean()

def epanechnikov(lmbda:float, point:float,
                 data_x:scipy.ndarray, data_y:scipy.ndarray,polynomial_order=0) -> float:
    t = scipy.absolute(data_x-point)/lmbda  # argment for D
    k = scipy.where(t <= 1, 0.75*(1-t), 0)    
    return predict(point,data_x,data_y,k,polynomial_order)

def gaussian(lmbda:float, point:float,
                 data_x:scipy.ndarray, data_y:scipy.ndarray,polynomial_order=0) -> float:
    k = (1/lmbda) * scipy.exp(
        -(scipy.absolute(data_x-point)*scipy.absolute(data_x-point)/(2*lmbda))) 
    return predict(point,data_x,data_y,k,polynomial_order)

def tricube(lmbda:float, point:float,
                 data_x:scipy.ndarray, data_y:scipy.ndarray,polynomial_order=0) -> float:
    t = scipy.absolute(data_x-point)/lmbda  # argment for D
    k = scipy.where(t <= 1, (70/81)*np.power((1-np.power(scipy.absolute(t),3)),3), 0)
    return predict(point,data_x,data_y,k,polynomial_order)


# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------
def plot_knn(ax,k,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample):
        x0 = xgrid[idx_x0]  # .5
        x_k = x_sample[scipy.argsort((x_sample-x0)*(x_sample-x0))[k-1:k]][0]
        mask = scipy.absolute(x_sample-x0) <= scipy.absolute(x0-x_k)
        set_ax(ax=ax, 
               y_estimated=scipy.array([knn(k, x, x_sample, y_sample) for x in xgrid]), 
               x_sample = x_sample,y_sample=y_sample,
               mask=mask, 
               title='Nearest-Neighbor Kernel',
               idx_x0=idx_x0, 
               show_true_y=show_true_y,show_estimated_y=show_estimated_y)

def plot_epanechnikov(ax,lmbda,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,polynomial_order=0):
    x0 = xgrid[idx_x0]  # .5
    set_ax(ax=ax, 
           y_estimated = scipy.array([epanechnikov(lmbda, x, x_sample, y_sample,polynomial_order)
                                  for x in xgrid]), 
           x_sample = x_sample,y_sample=y_sample,
           mask = scipy.absolute(x_sample-x0)/lmbda <= 1, 
           title='Epanechnikov Kernel',
           idx_x0=idx_x0,
           show_true_y=show_true_y,show_estimated_y=show_estimated_y) 

def plot_gaussian(ax,lmbda,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,polynomial_order=0):
    x0 = xgrid[idx_x0]
    set_ax(ax=ax, 
           y_estimated = scipy.array([gaussian(lmbda, x, x_sample, y_sample,polynomial_order)
                                  for x in xgrid]), 
           x_sample = x_sample,y_sample=y_sample,
           mask = np.array([True]*len(x_sample)),
           title='Gaussian Kernel',
           idx_x0=idx_x0,
           show_true_y=show_true_y,show_estimated_y=show_estimated_y) 

def plot_tricube(ax,lmbda,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,polynomial_order=0):
    x0 = xgrid[idx_x0]
    set_ax(ax=ax, 
           y_estimated = scipy.array([tricube(lmbda, x, x_sample, y_sample,polynomial_order)
                                  for x in xgrid]), 
           x_sample = x_sample,y_sample=y_sample,
           mask = scipy.absolute(x_sample-x0)/lmbda <= 1,
           title='Tricube Kernel',
           idx_x0=idx_x0,
           show_true_y=show_true_y, show_estimated_y=show_estimated_y)

def set_ax(ax, y_estimated, x_sample,y_sample, mask, title:str, idx_x0=500, show_true_y=True,show_estimated_y=True):
    x0 = xgrid[idx_x0]  
    if show_true_y:
        ax.plot(xgrid, y_true, color='C0', linewidth=2)
    if show_estimated_y:
        ax.plot(xgrid, y_estimated, color='C2')
        ax.plot((x0, x0), (min(y_sample), y_estimated[idx_x0]), 'o-', color='C3')
        ax.set_title(title)
    ax.plot(x_sample[mask], y_sample[mask],'o', color='C3', mfc='none')
    ax.plot(x_sample[~mask], y_sample[~mask],'o', color='C0', mfc='none')
   
def plot_two(right:str, left:str, 
             k_left:int, k_right:int,lmbda_left:float,lmbda_right:float,
             idx_x0=500, show_true_y=True,show_estimated_y=True, seed=42,
             x_sample=x_sample,y_sample=y_sample,pol_order_left=0,pol_order_right=0):    
    if seed!=42:
        x_sample, y_sample = draw_random_sample(seed)
    fig61 = plt.figure(61, figsize=(18, 10))
    ax1=fig61.add_subplot(1, 2, 1)
    plt.ylim(-1.3,2)
    plt.xlim(-0.2,1.2)
    ax2=fig61.add_subplot(1, 2, 2)
    plt.ylim(-1.3,2)
    plt.xlim(-0.2,1.2)
    
    if left == 'tricube':
        plot_tricube(ax1,lmbda_left,idx_x0,show_true_y, show_estimated_y,x_sample,y_sample,pol_order_left)
    elif left == 'epanechnikov':
        plot_epanechnikov(ax1,lmbda_left,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order_left)
    elif left == 'gaussian':
        plot_gaussian(ax1,lmbda_left,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order_left)
    elif left == 'knn':
        plot_knn(ax1,k_left,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample) 
    else: 
        raise ValueError('Unsupported left kernel name %s' %left)
        
    if right == 'tricube':
        plot_tricube(ax2,lmbda_right,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order_right)
    elif right == 'epanechnikov':
        plot_epanechnikov(ax2,lmbda_right,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order_right)
    elif right == 'gaussian':
        plot_gaussian(ax2,lmbda_right,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order_right)
    elif right == 'knn':
        plot_knn(ax2,k_right,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample) 
    else: 
        raise ValueError('Unsupported right kernel name %s' %right)
    
def plot_one(kernel:str, 
             k:int,lmbda,
             idx_x0=500, show_true_y=True,show_estimated_y=True, 
            seed = 42, x_sample=x_sample,y_sample=y_sample,pol_order=0):
    if seed!=42:
        x_sample, y_sample = draw_random_sample(seed)
    fig61 = plt.figure(61, figsize=(18, 10))
    ax=fig61.add_subplot(1, 2, 1)
    plt.ylim(-1.3,2)
    plt.xlim(-0.2,1.2)
    
    if kernel == 'tricube':
        plot_tricube(ax,lmbda,idx_x0,show_true_y, show_estimated_y,x_sample,y_sample,pol_order)
    elif kernel == 'epanechnikov':
        plot_epanechnikov(ax,lmbda,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order)
    elif kernel == 'gaussian':
        plot_gaussian(ax,lmbda,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample,pol_order)
    elif kernel == 'knn':
        plot_knn(ax,k,idx_x0,show_true_y,show_estimated_y,x_sample,y_sample) 
    else: 
        raise ValueError('Unsupported kernel name %s' %kernel)


def interact_plot_two(
            right = ['tricube','epanechnikov','gaussian','knn'],
            left = ['tricube','epanechnikov','gaussian','knn'],
            k_left=widgets.IntSlider(min=1,max=100,step=1,value=10),
            k_right=widgets.IntSlider(min=1,max=100,step=1,value=10),
            lmbda_left=(0.02,3.0,0.1),
            lmbda_right=(0.02,3.0,0.1),
            idx_x0=widgets.IntSlider(min=1,max=1000,step=1,value=500),
            show_true_y=False,
            show_estimated_y=False, 
            seed=widgets.IntSlider(min=1,max=42,step=1,value=42),
            x_sample=fixed(x_sample),y_sample=fixed(y_sample),
            pol_order_left=widgets.IntSlider(min=0,max=2,step=1,value=0),
            pol_order_right=widgets.IntSlider(min=0,max=2,step=1,value=0)):
    
    interact(plot_two, 
            right = right,
            left = left,
            k_left=k_left,
            k_right=k_right,
            lmbda_left=lmbda_left,
            lmbda_right=lmbda_right,
            idx_x0=idx_x0,
            show_true_y=show_true_y,
            show_estimated_y=show_estimated_y, 
            seed=seed,
            x_sample=x_sample,
            y_sample=y_sample,
            pol_order_left=pol_order_left,
            pol_order_right=pol_order_right)

def interact_plot_one(
            kernel = ['tricube','epanechnikov','gaussian','knn'],
            k=widgets.IntSlider(min=1,max=100,step=1,value=10),
            lmbda=(0.02,3.0,0.01),
            idx_x0=widgets.IntSlider(min=1,max=1000,step=1,value=500),
            show_true_y=False,
            show_estimated_y=False,
            seed=widgets.IntSlider(min=1,max=42,step=1,value=42),
            x_sample=fixed(x_sample),y_sample=fixed(y_sample),
            pol_order = widgets.IntSlider(min=0,max=2,step=1,value=0)):
    interact(plot_one, 
            kernel = kernel,
            k=k,
            lmbda=lmbda,
            idx_x0=idx_x0,
            show_true_y=show_true_y,
            show_estimated_y=show_estimated_y,
            seed=seed,
            x_sample=x_sample,y_sample=y_sample,
            pol_order = pol_order)