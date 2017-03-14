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
import matplotlib.pyplot as plt
import time 
from model.kinematics import *
from model.Simulator import KinematicActuator
from model.Simulator import BodySimulator
from model.GoalSelector import oscillator

np.set_printoptions(suppress=True, precision=5, linewidth=9999999)


#-------------------------------------------------------------------------

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

def test_touches():


    plt.ion()

    # init plot 
    fig = plt.figure("arm")
    ax = fig.add_subplot(111,aspect="equal")
    vertices = ax.scatter(0,0, alpha=0.1) 

    trals = 100
    stime = 100

    actuator = KinematicActuator()

    for k in xrange(trals):
        for t in xrange(stime):
            angles = oscillator(t, 30, 
                    np.random.rand(actuator.NUMBER_OF_JOINTS*2) )
            l_angles = angles[:actuator.NUMBER_OF_JOINTS] 
            r_angles = angles[actuator.NUMBER_OF_JOINTS:] 
            
            l_angles, r_angles = BodySimulator.rescale_angles(l_angles, r_angles) 
            actuator.set_angles(self.larm_angles, self.rarm_angles)

            body = np.vstack((actuator.l_position, actuator.r_position))
            
            vertices.set_offset(body)

            fig1.canvas.draw()
            plt.pause(0.01)




if __name__ == "__main__" :
    
    test_touches()
