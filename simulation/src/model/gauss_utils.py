##
## Copyright (c) 2018 Francesco Mannella. 
## 
## This file is part of sensorimotor-contingencies
## (see https://github.com/GOAL-Robots/CNRUPD_010618_sensorimotorcontingencies).
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
import numpy as np
import numpy.random as rnd
import time
import os


def clipped_exp(x):
    cx = np.clip(x, -700, 700)
    return np.exp(cx)

def pseudo_diag(rows, cols):

    bn = rows/float(cols)
    by_cols = True
    if bn < 1:
        by_cols = False
        bn = cols/float(rows)

    bn = int(bn)

    M = np.zeros([rows, cols])
    if by_cols == True :
        for x in xrange(cols):
            xx = x*bn
            M[xx:(xx+bn), x ] = 1
    else :
        for y in xrange(rows):
            yy = y*bn
            M[y, yy:(yy+bn)] = 1

    return M

#------------------------------------------------------------
def map1DND(x,nDim,nBin) :
    ''' Map an index in one dimension with a tuple of indices in 'nDim' dinemsions.


    :param x: Original 1-dimensional index
    :param nDim: Number of dimensions to map to
    :param nBin: length of the dimensions to map to
    :type x: int
    :type nDim: int
    :type nBin: list(int)

  '''

    idx_max = np.prod(nBin) -1

    if x > idx_max :
        raise(ValueError( "index is greater than Bin**nDim" ))

    idcs = np.zeros(nDim)
    for i in xrange(nDim) :
        idcs[nDim -1 -i] = x%nBin[nDim -1 -i]
        x /= nBin[nDim -1 -i]

    return idcs.astype("int")

def mapND1D(idcs,nBin) :
    ''' Map a tuple of indices in 'nDim' dimensions with an index in one dinemsion.

    :param x: Original 1-dimensional index
    :param nBin: length of the original dimensions
    :type x: int
    :type nBin: list(int)

    '''
    nDim = len(idcs)

    return int(sum( [  idcs[nDim -1 -x]*(nBin[x]**x)
        for x in xrange(nDim) ] ))

def grid(bins) :
    ''' Build a matrix of the indices of nodes on a multidimensional grid

    It takes a vector number-of-bins per dimensions ad stores a list of
    multidimensional indices  (ranging from 0:(bin_length-1) for each dimension)

    :param bins: gives the number of bins for each dimension
    :type bins: list(int)
    :returns: a matrix whose rows are the multidimensional indices
    :type: array((n_nodes, n_dims),int)

    Examples:

    > grid([2,3])  # asked for a 2X3 2-dimensional grid
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 0.,  2.],
           [ 1.,  0.],
           [ 1.,  1.],
           [ 1.,  2.]])

    > grid([2,3])  # asked for a 2X2X2 3-dimensional grid

    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  1.,  1.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  1.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  1.]])


    '''
    ret = np.array([])

    factor = np.arange(bins[0])

    if len(bins) > 1 :

        block = grid(bins[1:])
        n = len(block)

        blocks = []
        for factor_value in factor :

            factor_index = factor_value*np.ones([ n ,1])
            blocks.append(np.hstack( [ factor_index, block ] ))

        ret = np.vstack(blocks)

    else :
        ret = factor.reshape(int(bins[0]), 1)

    return ret

def scaled_grid(idcs, lims) :
    ''' Rescale grid to real ranges

    Takes indices builded with grid()
    and rescale each dimension with bounds from lims

    :param idcs: a matrix whose rows are the multidimensional indices
    :param lims: each row has (at least) the min and max values for that dimension
    :type idcs: array(((n_nodes, n_dims),int)
    :type lims: array(((n_dims,2),float)
    :returns: a matrix whose rows are the rescaled multidimensional indices
    :rtype: array((n_nodes, n_dims),float)

    Examples:
    > idcs = grid([2,3])  # asked for a 2X3 2-dimensional grid
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 0.,  2.],
           [ 1.,  0.],
           [ 1.,  1.],
           [ 1.,  2.]])

    > scaled_grid(idcs, [[0, 100], [-1, 1]])
    array([[   0.,   -1.],
           [   0.,    0.],
           [   0.,    1.],
           [ 100.,   -1.],
           [ 100.,    0.],
           [ 100.,    1.]])

    '''
    idcs = idcs.astype("int")
    scaled = np.zeros(idcs.shape)

    for i in xrange(len(lims)):

        x = np.linspace(*lims[i], num = (max(idcs[:,i])+1))
        scaled[:,i] += x[idcs[:,i]]

    return scaled


class MultidimensionalGaussianMaker(object) :
    ''' Build gaussians given a multidimensional space

        Example:

        > # make a description of a 2-dimensional
        > # 10x10 grid having
        > # x-bounds = [-1, 1] and y-bounds = [-2, 1]
        > lims = [[-1, 1, 10], [-2, 1, 10]]

        > # Build a gaussian maker with the above description
        > gm = MultidimensionalGaussianMaker(lims)

        > # make a gaussian with mu = [.5,0] and
        > # sigma = [.5,.3]
        > grid_value, grid_idcs = gm( mu = [.5,0], sigma = [.5,.3])

        > plot the gaussian with scatter
        > plt.scatter(grid_idcs[:,0], grid_idcs[:,1], s=100*grid_values)

    '''
    def __init__(self, lims) :
        '''
        :param lims:  each row has the min, max,and number-of-bins for that dimension
        :type lims: array(((n_dims,2),float)
        '''

        lims = np.array(lims)
        idcs = grid(lims[:,2])

        self.X = scaled_grid(idcs, lims[:,:2])
        self.vertices_number = self.X.shape[0]
        self.nDim = self.X.shape[1]

    def __call__(self, mu, sigma) :
        '''
        :param mu: the mean of the gaussian in the n-dimensional space
        :param sigma: the standard deviation of the gaussian in the n-dimensional space
        :type mu: array((n_dims), float)
        :type sigma: array((n_dims), float)
        :returns: a vector of values for each node and a matrix of multidimensional indices
        :rtype: tuple(array((n_nodes), float), array((n_nodes, n_dims), int) )
        '''

        mu = np.array(mu)
        sigma = np.array(sigma)

        e = (self.X - mu).T
        S = np.eye(self.nDim, self.nDim)*(1.0/sigma)
        y = clipped_exp( -np.diag(np.dot(e.T, np.dot(S,e) ) ))

        return (y, self.X)



class TwoDimensionalGaussianMaker(object) :
    ''' Fast build 2-dimensional goussians
    '''

    def __init__(self, lims) :
        '''
        :param lims: each row has the min and max values for that dimension
        :type lims: array(((n_dims,2),float)

        '''

        lims = np.vstack(lims)
        x = np.linspace(*lims[0])
        y = np.linspace(*lims[1])
        self.X, self.Y = np.meshgrid(x,y)

    def __call__(self, mu, sigma, theta=0) :

        '''
        :param mu: the mean of the gaussian in the 2-dimensional space
        :param sigma: the standard deviation of the gaussian in the 2-dimensional space
        :param theta: the angle of the axis of variance
                      of the gaussian in the 2-dimensional space
        :type mu: array((2), float)
        :type sigma: array((2), float)
        :type theta: float
        :returns: a vector of values for each node and a list with the X, and Y mashes
        :rtype: tuple(array((n_nodes), float), array((n_nodes, 2), int) )
        '''

        if np.isscalar(sigma) == True :
            sigma = [sigma,sigma]

        sx,sy = sigma
        mx,my = mu

        a = (np.cos(theta)**2)/(2*sx**2) +\
            (np.sin(theta)**2)/(2*sy**2);
        b = (-np.sin(2*theta))/(4*sx**2) +\
            (np.sin(2*theta))/(4*sy**2);
        c = (np.sin(theta)**2)/(2*sx**2) +\
            (np.cos(theta)**2)/(2*sy**2);

        res = clipped_exp(
                -a*(self.X-mx)**2
                -2*b*(self.X-mx)*(self.Y-my)
            -c*(self.Y-my)**2)

        return res.T.ravel(), [self.X, self.Y]

class OneDimensionalGaussianMaker(object) :
    ''' Fast build 1-dimensional goussians
    '''

    def __init__(self, lims) :

        self.x = np.linspace(*lims[0])

    def __call__(self, mu, sigma) :
        '''
        :param mu: the mean of the gaussian in the 1-dimensional space
        :param sigma: the standard deviation of the gaussian in the 1-dimensional space
        :type mu: array((1), float)
        :type sigma: array((1), float)
        :returns: a vector of values for each node and the vector of x points
        :rtype: tuple(array((n_nodes), float), array((n_nodes, 1), int) )
        '''

        return clipped_exp((-(self.x-mu)**2)/(sigma**2)), self.x

class OptimizedGaussianMaker(object) :
    ''' Wrapper to the 2-D, 1-D, n-D gaussian makers
    '''

    def __init__(self, lims) :
        '''
        :param lims: each row has the min, max, and number-of-bins for that dimension.
                     rows in lims decrete which kind of optimizer (1-D, 2-D n-D) will be used
        :type lims: array(((n_dims,3),float)
        '''

        L = len(lims)
        self.gm = None
        if L == 1 :
            self.gm = OneDimensionalGaussianMaker(lims)
        elif L == 2:
            self.gm = TwoDimensionalGaussianMaker(lims)
        else:
            self.gm = MultidimensionalGaussianMaker(lims)

    def __call__(self, mu, sigma) :
        '''
        :param mu: the mean of the gaussian in the 1-dimensional space
        :param sigma: the standard deviation of the gaussian in the 1-dimensional space
        :type mu: array((1), float)
        :type sigma: array((1), float)

        :returns container of value ad container of indices ():
        :rtype: depends on the actual optimizer used
        '''
        return self.gm(mu, sigma)

def gauss2d_oriented(x,y,m1,m2,std_x, std_y, theta) :


    a = (np.cos(theta)**2)/(2*std_x**2) +\
        (np.sin(theta)**2)/(2*std_y**2);
    b = (-np.sin(2*theta))/(4*std_x**2) +\
        (np.sin(2*theta))/(4*std_y**2);
    c = (np.sin(theta)**2)/(2*std_x**2) +\
        (np.cos(theta)**2)/(2*std_y**2);

    return clipped_exp( -a*(x-m1)**2 -2*b*(x-m1)*(y-m2) -c*(y-m2)**2)


if __name__ == "__main__" :
    pass
