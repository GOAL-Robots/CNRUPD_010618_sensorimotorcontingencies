#!/usr/bin/env python
"""

The MIT License (MIT)

Copyright (c) 2015 Francesco Mannella <francesco.mannella@gmail.com> 

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from __future__ import division

import numpy as np
import numpy.random as rnd

from scipy.ndimage import gaussian_filter1d
import scipy.optimize  
import time 
np.set_printoptions(suppress=True, precision=5, linewidth=9999999)


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# UTILS ---------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# cross product
def cross(a, b): return a[0]*b[1] - a[1]*b[0]

def get_angle(v1,v2) :
    """
    Calculate the angle between two vectors

    see https://goo.gl/Uq0lw

    v1  (array):   first vector
    v2  (array):   second vector
    """

    if (np.linalg.norm(v1)*np.linalg.norm(v2)) != 0 :     
        cosangle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        cosangle = np.maximum(-1,np.minimum(1, cosangle))
        angle = np.arccos(cosangle)   
        if np.cross(v1,v2) < 0 :
            angle = 2*np.pi - angle  
        return angle

    return None


# Manages distances on a polynomial chain
class Polychain(object) :

    # versor defining the absoliute reference frame for angles
    HVERSOR = np.array([1,0])

    def set_chain(self, chain) :
        '''
         Configure the poly chain with a given list of points
         
         :param chain   the polygonal chain
         :type chain    a list of (x,y) points 
        '''

        # ensure it is a numpy array
        self.chain = np.array(chain[:])
        # the length of the chain (number of vertices)
        self.vertices_number = len(self.chain)
        # the chain must be greater than one point
        if self.vertices_number < 2 :
            raise ValueError('Polychain initialized with only one point. Minimum required is two.')
       
        # calculate segments lengths
        self.segment_lengths = [ 
                np.linalg.norm( self.chain[x] -  self.chain[x-1] ) \
                for x in xrange(1, self.vertices_number) ]

        # calculate angles at the vertices
        self.segment_angles = []
        for x in xrange(1, self.vertices_number ) :
            # define ab segment. (the first ab segment is the versor defining horizontal axis)
            ab = self.HVERSOR if x == 1 else self.chain[x-1] - self.chain[x-2]
            # define bc segment
            bc = self.chain[x] - self.chain[x-1]
            # compute abVbc angle and add it to segment_angles 
            self.segment_angles.append( get_angle(ab, bc) )

    def autocollision(self, epsilon = 0.1, is_set_collinear=False, debug=False):
        '''
        Detect if the chain is self-colliding    
        by reading intersections between all the segments of the chain

        from http://stackoverflow.com/a/565282/4209308
        
        q+s = sq   p+r = rp
            \       /
             \     /
              \   /
               \ /
          p+t*r \ q+u*s  
               / \
              /   \
             /     \
            p       q

        
        :param  epsilon             a trheshold of sensitivity of 
                                    each point of the chain (how much they 
                                    must be close one another to "touch") 

        :param  is_set_collinear    further test for collinearity (segments are 
                                    parallel and non-intersecting)     
        '''

        self.intersect = None
        (start, end) = self.chain[[1,-1]]

        n = len(self.chain)
        rng = range(1,n)        
        
        # iterate over all combinations of pairs of segments
        # of the polychain and for each pair compute intersection

        for x in rng :
            
            # points of the first segment
            p = self.chain[x-1]    # start point
            rp = self.chain[x]    # end point
            r = rp - p    # end point relative to the start point
            
            for y in rng :

                # points of the second segment
                q = self.chain[y-1]    # start point
                sq = self.chain[y]    # end point 
                s = sq - q    # end point relative to the start point                 
                
                # test if start or end points overlap
                junction = np.all(p == sq) or np.all(q == rp) 
                
                # only compute collisions if end points are not junktions
                if x!=y and not junction :

                    # cross product of the relative end points.
                    # if rxs = 0 the two segments are parallels
                    rxs = cross(r,s)
                    rxs_zero = abs(rxs) < epsilon

                    
                    qpxr = cross(q-p, r)
                    qpxs = cross(q-p, s)

                    # segments are parallel
                    if rxs_zero :
                         if is_set_collinear:
                             
                             # segments are collinear   
                             if abs(qpxr) < epsilon : 
                                 
                                t0 = np.dot(q-p, r/np.dot(r,r))
                                t1 = t0 + np.dot(s, r/np.dot(r,r))
                                
                                mint = min(t0, t1)
                                maxt = max(t0, t1)

                                # segments overlap
                                if 0<maxt<1 or 0<mint<1  :
                                    return (True,p,rp,r,q,sq,s) if debug else True


                    # segments are not parallel
                    if not rxs_zero :
                        t = qpxs / rxs  
                        u = qpxr / rxs   
                        
                        int1 = p+t*r
                        int2 = q+u*s
                        
                        # segments intersect  
                        if 0 <= t <= 1 and 0 <= u <= 1: 
                            self.intersect = int1
                            return (True,p,rp,int1,q,sq,int2) if debug else True

        return (False,[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]) if debug else False
                
    def isPointInChain(self, point, epsilon = 0.1 ) :
        '''
            find out if a point belongs to the chain. 
            return:     a list of distances correspoding to the the line 
                        intersection with that point. Empty list if
                        the point does not belong to the chain
        '''

        distances = []
        c = array(point)
        for x in xrange(1,len(self.chain) ) :
            
            a = self.chain[x-1]
            b = self.chain[x]
            
            # check if the  point is within the same line
            if np.all(c!=a) and np.all(c!=b) :
                if np.linalg.norm(np.cross(b-a, c-a)) < epsilon :
                    
                    abac = np.dot(b-a, c-a)
                    ab = np.dot(b-a, b-a)
                    if 0 <= abac <= ab :

                        distance = np.sum(self.segment_lengths[:(x-1)])
                        distance += np.linalg.norm(point - self.chain[x-1])
                        distance = distance/sum(self.segment_lengths)

                        distances.append( distance )


        return distances
      
    def get_point(self, distance) :
        '''
        get a point in the 2D space given a 
        distance from the first point of the chain 
        '''

        if distance > 1 :
            raise ValueError('distance must be a proportion of the polyline length (0,1)')

        if distance == 0.0 : 
            return self.chain[0]

        if distance == 1.0 : 
            return self.chain[-1]

        scaled_distance = sum(self.segment_lengths)*distance
        cumulated_length = 0
            
        for segment_index in xrange(self.vertices_number-1) :
            current_segment_length = self.segment_lengths[segment_index]
            if cumulated_length <= scaled_distance <= \
                    (cumulated_length + current_segment_length) :
                break 
            cumulated_length += self.segment_lengths[segment_index]

        last_segment_distance = scaled_distance - cumulated_length
        
        last_segment_x = last_segment_distance*np.cos( \
                sum(self.segment_angles[:(segment_index+1)]) )
        last_segment_y = last_segment_distance*np.sin( \
                sum(self.segment_angles[:(segment_index+1)]) )

        return self.chain[segment_index] + (last_segment_x, last_segment_y)
        

    def get_dense_chain(self, density) :

        
        dense_chain = []
        segment_number = density - 1 
        
        dense_chain.append( self.get_point(0) )
        
        for x in xrange( segment_number ) :
            dense_chain.append(self.get_point( (1+x)/float(segment_number) ))

        return np.vstack(dense_chain)

    def get_length(self) :
        '''
        return: the length of the current polyline
        '''
        return sum(self.segment_lengths)





#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ARM -----------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class PID(object) :

    def __init__(self, n=1, dt=0.1, Kp=0.1, Ki=0.9, Kd=0.001 ):
       
        self.n = n
        self.dt = dt

        self.previous_error = np.zeros(n)
        self.integral = np.zeros(n)
        self.derivative = np.zeros(n)
        self.setpoint = np.zeros(n)
        self.output = np.zeros(n)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def reset(self):
        n = self.n
        self.previous_error = np.zeros(n)
        self.integral = np.zeros(n)
        self.derivative = np.zeros(n)
        self.setpoint = np.zeros(n)
        self.output = np.zeros(n)

    def step(self, measured_value, setpoint=None):
        
        if setpoint is not None:
            self.setpoint = np.array(setpoint)

        error = setpoint - measured_value
        self.integral =  self.integral + error*self.dt
        self.derivative = (error - self.previous_error)/self.dt
        self.output = self.Kp*error + \
                self.Ki*self.integral + \
                self.Kd*self.derivative
        
        self.previous_error = error

        return self.output

class Arm(object):
    """
    
    Kinematics of a number_of_joint-degrees-of-freedom 
    2-dimensional arm.
    
    Given the increment of joint angles calculate
    the current positions of the edges of the
    arm segments.

    """

    def __init__(self,
            number_of_joint = 3,
            joint_lims = None, 
            segment_lengths = None,
            origin = [0,0],
            mirror = False
            ):
        """
        number_of_joint:     (int)    number of joints
        joint_angles       (list):    initial joint angles
        joint_lims         (list):    joint angles limits 
        segment_lengths    (list):    length of arm segmens
        origin             (list):    origin coords of arm
        """

        self.mirror = mirror
        self.number_of_joint = number_of_joint
        
        # initialize lengths
        if segment_lengths is None:
            segment_lengths = np.ones(number_of_joint)
        self.segment_lengths = np.array(segment_lengths)
       
        # initialize limits   
        if joint_lims is None:
            joint_lims = vstack([-np.ones(number_of_joint)*
                np.pi,ones(number_of_joint)*np.pi]).T
        self.joint_lims = np.array(joint_lims)
       
        # set origin coords   
        self.origin = np.array(origin)
        
    def get_joint_positions(self,  joint_angles  ):
        """    
        Finds the (x, y) coordinates
        of each joint   
        joint_angles   (vector):    current angles of the joints    
        return          (array):    'number of joint' [x,y] coordinates   
        """ 


        # current angles
        res_joint_angles = joint_angles.copy() 

        # detect limits
        maskminus= res_joint_angles > self.joint_lims[:,0]
        maskplus = res_joint_angles < self.joint_lims[:,1]
  
        res_joint_angles =  res_joint_angles*(maskplus*maskminus) 
        res_joint_angles += self.joint_lims[:,0]*(np.logical_not(maskminus) )
        res_joint_angles += self.joint_lims[:,1]*(np.logical_not(maskplus) )
 
        # mirror
        if self.mirror :
            res_joint_angles = -res_joint_angles
            res_joint_angles[0] += np.pi 
 
        # calculate x coords of arm edges.
        # the x-axis position of each edge is the 
        # sum of its projection on the x-axis
        # and all the projections of the 
        # previous edges 
        x = np.array([  
                    sum([ 
                            self.segment_lengths[k] *
                            np.cos( (res_joint_angles[:(k+1)]).sum() ) 
                            for k in range(j+1) 
                        ])
                        for j in range(self.number_of_joint) 
                  ])
        
        # translate to the x origin 
        x = np.hstack([self.origin[0], x+self.origin[0]])

        # calculate y coords of arm edges.
        # the y-axis position of each edge is the 
        # sum of its projection on the x-axis
        # and all the projections of the 
        # previous edges 
        y = np.array([  
            sum([ 
                    self.segment_lengths[k] *
                    np.sin( (res_joint_angles[:(k+1)]).sum() ) 
                    for k in range(j+1) 
                ])
                for j in range(self.number_of_joint) 
            ])
        
        # translate to the y origin 
        y = np.hstack([self.origin[1], y+self.origin[1]])

        pos = np.array([x, y]).T
 
        return (pos, res_joint_angles)
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# TEST ----------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


def oscillator(x, scale, params) :
    '''
    :param  x       list of timesteps
    :param  scale   scaling factor for the frequency 
    :param  params  list parameters' pairs (for each pair a trajectory is produced)
    '''
    
    x = np.array(x) # timeseries
    params= np.array(params)
    pfreq = params[:int(params.size/2)]
    pph = params[int(params.size/2):]
    x = np.outer(x, np.ones(pfreq.shape))

    return np.pi*np.cos(np.pi*(pfreq*x/scale-10*pph))

def test_simple_collision():
    import matplotlib.pyplot as plt

    polychain = Polychain()

    a = [0,0]
    b = [0,1]
    c = [1.5,1]
    c1 = [1.9,1]
    dd = [0.9,0]
    d = [1.6,0.0]
    e = [.5,-0.1]
    f = [1.5,0.0]

    rbt = np.vstack((dd, a, b, c,  d, f))
    
    polychain.set_chain(rbt)

    (if_coll,p,r,t,q,s,u) = polychain.autocollision(
                epsilon = 0.1, is_set_collinear=True, debug=True)

    print if_coll
    plt.plot(*rbt.T)
    plt.plot(*np.vstack([p,r]).T, color="green", lw=3)
    plt.plot(*np.vstack([q,s]).T, color="red",lw=2)
    plt.xlim([-1,2])
    plt.ylim([-1,2])
    plt.show()

def test_collision(plot=True):

    import matplotlib.pyplot as plt
    plt.ion()

    polychain = Polychain()

    larm = Arm(
            number_of_joint = 3,    # 3 joints 
            origin = [-1.5,0.0], # origin at (1.0 , 0.5)
            segment_lengths = np.array([1,1,1]),
            joint_lims = [ 
                [0, np.pi*0.9],    # first joint limits                   
                [0, np.pi*0.6],    # second joint limits             
                [0, np.pi*0.6],     # third joint limits
                ],  
            mirror=True
            )
    
    rarm = Arm(
            number_of_joint = 3,    # 3 joints 
            origin = [1.5,0.0], # origin at (1.0 , 0.5)
            segment_lengths = np.array([1,1,1]),
            joint_lims = [ 
                [0, np.pi*0.9],    # first joint limits                   
                [0, np.pi*0.6],    # second joint limits             
                [0, np.pi*0.6],     # third joint limits
                ] 
            )

    
    stime = 200
    xline = np.arange(stime)
    trials = 400 
    n_sensors = 30
    langle =  np.vstack([ oscillator(xline,30,np.random.rand(6)) for i in range(trials) ])
    rangle =  np.vstack([ oscillator(xline,30,np.random.rand(6)) for i in range(trials) ])
    lpos,langle[0,:] = larm.get_joint_positions(langle[0,:])
    rpos,rangle[0,:] = rarm.get_joint_positions(rangle[0,:])
    rbt = np.vstack((lpos[::-1],rpos))
    
    
    if plot == True :
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect="equal")
        robot_img, = plt.plot(*rbt.T)
        robot_jts = plt.scatter(*rbt.T)
        prpl, = plt.plot(*(np.ones([2,2])*10000).T, lw=3, c="red")
        prpp = plt.scatter(*(np.ones([2,2])*10000).T, s=60, c="red")
        qsql, = plt.plot(*(np.ones([2,2])*10000).T, lw=3, c="green")
        qsqp = plt.scatter(*(np.ones([2,2])*10000).T, s=60, c="green")
        prtp = plt.scatter(*(np.ones([2,2])*10000).T, s=120, color="yellow")
        plt.xlim([-4,4])
        plt.ylim([-3,3])

    for x in range(stime*trials):
        
        rbt = np.vstack((lpos[::-1],rpos))
        polychain.set_chain(rbt)
        

        (if_coll,p,r,t,q,s,u) = polychain.autocollision(
                epsilon = 0.1, is_set_collinear=True, debug=True)
        

        if plot == True:
            robot_img.set_data(rbt.T)
            robot_jts.set_offsets(rbt)
        
            if if_coll==True :
                prpp.set_offsets( np.vstack([p,r]) )
                qsqp.set_offsets( np.vstack([q,s]) )
                prtp.set_offsets( np.vstack([t,u]) )
                prpl.set_data( *np.vstack([p,r]).T )
                qsql.set_data( *np.vstack([q,s]).T )
            else :
                prpp.set_offsets( np.vstack([p,r])*1e10 )
                qsqp.set_offsets( np.vstack([q,s])*1e10 )
                prtp.set_offsets( np.vstack([p,r])*1e10 )
                prpl.set_data( *np.vstack([p,r]).T*1e10 )
                qsql.set_data( *np.vstack([q,s]).T*1e10 )
        else:
            print 

        fig.canvas.draw()    
        plt.pause(0.05) 
    
        lpos,langle[x,:] = larm.get_joint_positions(langle[x,:])
        rpos,rangle[x,:] = rarm.get_joint_positions(rangle[x,:])

          
    raw_input()    


def test_rots():

    import matplotlib.pyplot as plt
    plt.ion()
  
    arm = Arm(
            number_of_joint = 4,    # 3 joints 
            origin = [0.0,0.0], # origin at (1.0 , 0.5)
            segment_lengths = np.array([2,2,2,2])/2.,
            joint_lims = [ 
                [0, np.pi],    # first joint limits                   
                [0, np.pi],    # second joint limits             
                [0, np.pi],     # third joint limits
                [0, np.pi]     # third joint limits
                ],  
            mirror=True
            )
    
    angle = np.zeros(4)  
    pos,angle = arm.get_joint_positions(angle)
    poly = Polychain()
    poly.set_chain(pos)
    point = poly.get_point(0.75)
 
    
    # init plot 
    fig = plt.figure("arm")
    ax = fig.add_subplot(111,aspect="equal")
    segments, = ax.plot(*pos.T)    # plot arm segments
    edges = ax.scatter(*pos.T) # plot arm edges
    external_point = plt.scatter(*point, s= 30, c="r") # plot arm edges
    dense_points = plt.scatter(*np.zeros([2,25]), s= 20, c=[1,1,0]) # plot arm edges
   
    xl = [-8,2]    # x-axis limits
    yl = [-2,8]    #y-axis limits 
    ax.plot(xl, [0,0], c = "black", linestyle = "--")    # plot x-axis
    ax.plot([0,0], yl, c = "black", linestyle = "--")    # plot y-axis     
    ax.set_xlim(xl)
    ax.set_ylim(yl) 

    # iterate over 100 timesteps
    for t in range(200):
      
        # set a random gaussian increment for each joint
        angle = np.ones(4)*(t*((2*np.pi)/400.0)) 
        # calculate current position given the increment
        pos, angle = arm.get_joint_positions(angle)
        poly.set_chain(pos)
        
        collision = poly.autocollision(is_set_collinear=True)
        point = poly.get_point(0.75)
        dense = poly.get_dense_chain(10)

        # update plot
        segments.set_data(*pos.T)
        edges.set_offsets(pos)
        external_point.set_offsets([point])
        dense_points.set_offsets(dense)
        
        if collision:
            segments.set_color('red')
            segments.set_linewidth(2)
        else:
            segments.set_color('blue')
            segments.set_linewidth(1)
        #external_point.set_offsets(point)
        #dense_points.set_offsets(dense)
        fig.canvas.draw()
        raw_input()

    raw_input()


if __name__ == "__main__" :
    
    test_collision()
