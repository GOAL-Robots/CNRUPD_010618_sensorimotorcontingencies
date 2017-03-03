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

    # floating point precision
    HVERSOR = np.array([1,0])

    def set_chain(self, chain) :
        '''
         :param chain   the polygonal chain
         :type chain    a list of (x,y) points 
        '''

        # ensure it is a numpy array
        self.chain = np.array(chain[:])
        # the length of the chain (number of vertices)
        self.ln = len(self.chain)
        # the chain must be greater than one point
        if self.ln < 2 :
            raise ValueError('Polychain initialized with only one point. Minimum required is two.')
       
        # calculate segments lengths
        self.seg_lens = [ 
                np.linalg.norm( self.chain[x] -  self.chain[x-1] ) \
                for x in xrange(1, self.ln) ]

        # calculate angles at the vertices
        self.seg_angles = []
        for x in xrange(1, self.ln ) :
            if x == 1 :
                ab = self.HVERSOR
            else :
                ab = self.chain[x-1] - self.chain[x-2]
            bc = self.chain[x] - self.chain[x-1]
            self.seg_angles.append(get_angle(ab, bc))

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
        rng = np.hstack((range(1,n), range(n-2,0,-1) ))        
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
                not_junction = np.all(p != sq) and np.all(q != rp) 

                # only compute collisions if end points are not junktions
                if x!=y and not_junction :

                    # cross product of the relative end points.
                    # if rxs = 0 the two segments are parallels
                    rxs = cross(r,s)
                    rxs_zero = abs(rxs) < epsilon

                    
                    qpxr = np.linalg.norm(cross(q-p, r))
                    qpxs = np.linalg.norm(cross(q-p, s))

                    # segments are parallel
                    if rxs_zero :
                         if is_set_collinear:
                             # segments are collinear
                             if qpxr < epsilon : 
                                t0 = np.dot(q-p,r)/np.dot(r,r)
                                t1 = t0 + np.dot(s,r)/np.dot(r,r)
                                
                                mint = min(t0, t1)
                                maxt = max(t0, t1)
                                # segments overlap
                                if (mint > (0+epsilon) and mint < (1+epsilon)) \
                                        or (maxt >  (0+epsilon) and maxt < (1-epsilon)) \
                                        or (mint <= (0+epsilon) and maxt >= (1-epsilon)):
                                    return (True,p,rp,r,q,sq,s) if debug else True


                    # segments are not parallel
                    if not rxs_zero :
                        t = qpxs / rxs  
                        u = qpxr / rxs   
                         
                        # segments intersect 
                        int1 = p+t*r
                        int2 = q+u*s

                        if np.linalg.norm(int1 - int2) < epsilon  and \
                                np.linalg.norm(t*r)<=np.linalg.norm(rp - p) and \
                                np.linalg.norm(u*s)<=np.linalg.norm(sq - q):
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

                        distance = np.sum(self.seg_lens[:(x-1)])
                        distance += np.linalg.norm(point - self.chain[x-1])
                        distance = distance/sum(self.seg_lens)

                        distances.append( distance )


        return distances
      
    def get_point(self, distance) :
        '''
        get a point in the 2D space given a 
        distance from the first point of the chain 
        '''

        if distance > 1 :
            raise ValueError('distance must be a proportion of the polyline length (0,1)')

        distance = sum(self.seg_lens)*distance
        cum_ln = 0
            
        for l in xrange(self.ln-1) :
            s_ln = self.seg_lens[l]
            if cum_ln <= distance <= cum_ln+s_ln :
                break 
            cum_ln += self.seg_lens[l]

        rel_ln = distance - cum_ln
        
        return self.chain[l] + \
                ( rel_ln*np.cos( sum(self.seg_angles[:(l+1)]) ), \
                    rel_ln*np.sin( sum(self.seg_angles[:(l+1)]) ) )

        return -1

    def get_dense_chain(self, density) :

        tot_len = self.get_length()
        curr_len = 0
        dense_chain = []
        points = density
        dense_chain.append(self.get_point( 0 ))
        for x in xrange( density ) :
            dense_chain.append(self.get_point( (1+x)/float(density+1) ))

        dense_chain.append(self.get_point( 1 ))

        return np.vstack(dense_chain)

    def get_length(self) :
        '''
        return: the length of the current polyline
        '''
        return sum(self.seg_lens)





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
                pi,ones(number_of_joint)*pi]).T
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
        
        # trabslate to the x origin 
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

def get_touch(body_tokens, chain, num_touch_sensors=30):
    '''
    Build the current touch retina.
    All body joints are viewed as points in a 1-dimensional space.
    At each joint position a 2D gaussian is computed with standard deviation
    on the vertical axis defined by the amplitude of the 
    joint angle times self.touch_retina_sigma[1] and 
    standard deviation on the orizontal axis defined by the amplitude of 
    self.sensors = self.chain.get_dense_chain(self.num_touch_sensors)   
    All gaussians are summed point-to-point to obtain a mixture.

    :param  angle_tokens    list of lists of angles of all body parts' joints
    :type   angle_token     listo f list of floats - [ [ja0, ..., jan], ... ]
    
    :return a 2D array 
    :rtype  np.array(self.size, dtype=float ) 
    
    ''' 
    
    #### compute touches

    # get the chain of the whole body (a single array of 2D points )
    bts = np.vstack(body_tokens)
    chain.set_chain( bts )

    # update sensors
    sensors = chain.get_dense_chain(num_touch_sensors)   

    # init the vector of touch measures
    sensors_n = len(sensors)
    touches = np.zeros(sensors_n)
    sensor_activations = [] 

    # read contact of the two edges (hands) with the rest of the points 
    for x,sensor  in zip( [0,sensors_n-1], [ sensors[0], sensors[-1] ] ):

        # for heach edge iterate over the other points
        for y,point in enumerate(sensors):        
            # do not count the edge with itself (it would be autotouch!!)
            if abs(y-x)>num_touch_sensors/4:
                
                # touch is measured as a radial basis of the distance from the sensor
                stouch = \
                        np.exp(-((np.linalg.norm(point - sensor))**2)/ \
                        (2*0.1**2)  )
                sensor_activations.append([x,y,stouch])

                touches[y] += stouch  

    sensor_activations = np.vstack(sensor_activations)

    return touches, sensor_activations



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

def test_touch():

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
    data = np.zeros([stime*trials, 16])
    data_sensors = np.zeros([stime*trials, n_sensors])
    data_sensor_activations = [] 
    langle =  np.vstack([ oscillator(xline,10,np.random.rand(6))*np.pi*1.5 for i in range(trials) ])
    rangle =  np.vstack([ oscillator(xline,10,np.random.rand(6))*np.pi*1.5 for i in range(trials) ])

    for x in range(stime*trials):
        
        lpos,langle[x,:] = larm.get_joint_positions(langle[x,:])
        rpos,rangle[x,:] = rarm.get_joint_positions(rangle[x,:])
        
        sensor_act_idcs = range(x*n_sensors, (x+1)*n_sensors)
        data_sensors[x,:], sensor_activations = \
                get_touch( np.vstack((lpos[::-1], rpos)) , \
                polychain, num_touch_sensors=n_sensors-2)
        
        sensor_activations = np.vstack([x*np.ones(len(sensor_activations)), \
                sensor_activations.T]).T
        data_sensor_activations.append(sensor_activations)
        data[x,:] = np.hstack((lpos.ravel(), rpos.ravel()))
        print x
    
    data_sensor_activations = np.vstack(data_sensor_activations)
    
    np.savetxt("/tmp/data_sensor_activations", data_sensor_activations)

    plt.figure()
     
    plt.scatter(*data[:,6:8].T, s=6,lw = 0, color="blue", alpha=.1)
    plt.scatter(*data[:,14:16].T, s=6,lw = 0, color="blue", alpha=.1)
  
    plt.scatter(*data[:,4:6].T, s=6,lw = 0, color="green", alpha=.1)
    plt.scatter(*data[:,12:14].T, s=6,lw = 0, color="green", alpha=.1)
     
    plt.scatter(*data[:,2:4].T, s=6,lw = 0, color="red", alpha=.1)
    plt.scatter(*data[:,10:12].T, s=6,lw = 0, color="red", alpha=.1)
 
    plt.ylim([-7,7])
    plt.xlim([-7,7])

    plt.figure()
    
    plt.bar(np.arange(n_sensors)+0.9, data_sensors.std(0), color="red", width=0.1)
    plt.bar(np.arange(n_sensors)+0.6, data_sensors.mean(0), color="blue", width=0.8)

    plt.xlim([0,32])
    
    raw_input()
   
    return data_sensor_activations

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
    langle =  np.vstack([ oscillator(xline,15,np.random.rand(6)) for i in range(trials) ])
    rangle =  np.vstack([ oscillator(xline,15,np.random.rand(6)) for i in range(trials) ])
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
        lpos,langle[x,:] = larm.get_joint_positions(langle[x,:])
        rpos,rangle[x,:] = rarm.get_joint_positions(rangle[x,:])
        rbt = np.vstack((lpos[::-1],rpos))
        polychain.set_chain(rbt)
        
        robot_img.set_data(rbt.T)
        robot_jts.set_offsets(rbt)
        fig.canvas.draw()    
        
        (if_coll,p,r,t,q,s,u) = polychain.autocollision(
<<<<<<< HEAD
                epsilon = 0.1, is_set_collinear=True, debug=True)
        

        if plot == True:
            if if_coll==True :
                prpp.set_offsets( np.vstack([p,r]) )
                qsqp.set_offsets( np.vstack([q,s]) )
                prtp.set_offsets( np.vstack([t,u]) )
                prpl.set_data( *np.vstack([p,r]).T )
                qsql.set_data( *np.vstack([q,s]).T )
                print t,u
            else :
                prpp.set_offsets( np.vstack([p,r])*1e10 )
                qsqp.set_offsets( np.vstack([q,s])*1e10 )
                prtp.set_offsets( np.vstack([p,r])*1e10 )
                prpl.set_data( *np.vstack([p,r]).T*1e10 )
                qsql.set_data( *np.vstack([q,s]).T*1e10 )
        else:
            print 
        
=======
                epsilon = 0.01, is_set_collinear=True, debug=True)
        

        if plot == True:
            if if_coll==True :
                prpp.set_offsets( np.vstack([p,r]) )
                qsqp.set_offsets( np.vstack([q,s]) )
                prtp.set_offsets( np.vstack([t,u]) )
                prpl.set_data( *np.vstack([p,r]).T )
                qsql.set_data( *np.vstack([q,s]).T )
                print t,u
            else :
                prpp.set_offsets( np.vstack([p,r])*1e10 )
                qsqp.set_offsets( np.vstack([q,s])*1e10 )
                prtp.set_offsets( np.vstack([p,r])*1e10 )
                prpl.set_data( *np.vstack([p,r]).T*1e10 )
                qsql.set_data( *np.vstack([q,s]).T*1e10 )
        else:
            print 
        
>>>>>>> mke_no_learning
        # plt.pause(0.1)

    raw_input()    


def test_rots():

    import matplotlib.pyplot as plt
    plt.ion()
  
    arm = Arm(
            number_of_joint = 6,    # 3 joints 
            origin = [0.0,0.0], # origin at (1.0 , 0.5)
            segment_lengths = np.array([1,1,3,1,1,2])/2.,
            joint_lims = [ 
                [0, np.pi],    # first joint limits                   
                [0, np.pi],    # second joint limits             
                [0, np.pi],     # third joint limits
                [0, np.pi],     # third joint limits
                [0, np.pi],     # third joint limits
                [0, np.pi]     # third joint limits
                ],  
            mirror=True
            )
    
    angle = np.zeros(6)  
    pos,angle = arm.get_joint_positions(angle)
    poly = Polychain()
    poly.set_chain(pos)
    point = poly.get_point(0.75)
 
    
    # init plot 
    fig = plt.figure("arm")
    ax = fig.add_subplot(111,aspect="equal")
    segments, = ax.plot(*pos.T)    # plot arm segments
    edges = ax.scatter(*pos.T) # plot arm edges
    xl = [-8,8]    # x-axis limits
    yl = [-8,8]    #y-axis limits
    ax.plot(xl, [0,0], c = "black", linestyle = "--")    # plot x-axis
    ax.plot([0,0], yl, c = "black", linestyle = "--")    # plot y-axis     
    external_point = plt.scatter(*point, s= 30, c="r") # plot arm edges
    dense_points = plt.scatter(*np.zeros([2,25]), s= 20, c=[1,1,0]) # plot arm edges
    plt.xlim(xl)
    plt.ylim(yl) 

    # iterate over 100 timesteps
    for t in range(200):
      
        # set a random gaussian increment for each joint
        angle = np.ones(6)*(t*((2*np.pi)/400.0)) 
        angle[5] = 0 
        # calculate current position given the increment
        pos, angle = arm.get_joint_positions(angle)
        poly.set_chain(pos)
        
        collision = poly.autocollision(is_set_collinear=True)
        point = poly.get_point(0.75)

        dense = poly.get_dense_chain(6)

        # update plot
        segments.set_data(*pos.T)
        if collision:
            segments.set_color('red')
            segments.set_linewidth(2)
        else:
            segments.set_color('blue')
            segments.set_linewidth(1)
        edges.set_offsets(pos)
        #external_point.set_offsets(point)
        #dense_points.set_offsets(dense)
        fig.canvas.draw()
        raw_input()

    raw_input()


if __name__ == "__main__" :
    
    test_collision()
