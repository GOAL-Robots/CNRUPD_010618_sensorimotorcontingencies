#!/usr/bin/env python
"""

The MIT License (MIT)

Copyright (c) 2016 Francesco Mannella <francesco.mannella@gmail.com>

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
np.set_printoptions(suppress=True, precision=5, linewidth=9999999)


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# UTILS ---------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


def cross(a, b):
    """
    Calculate the cross product between two vectors in a 2D space
    (redefined for speed)

    :param  v1  first vector
    :param  v2  second vector

    """

    return a[0]*b[1] - a[1]*b[0]


def get_angle(v1,v2) :
    """
    Calculate the angle between two vectors in a 2D space

    see https://goo.gl/Uq0lw

    :param  v1  first 2D edge
    :param  v2  second 2D edge
    :type   v1  a pair of double
    :type   v2  a pair of double

    :rtype   float
    """

    # must be both segments (not points)
    if (np.linalg.norm(v1)*np.linalg.norm(v2)) != 0 :
        cosangle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        cosangle = np.maximum(-1,np.minimum(1, cosangle))
        angle = np.arccos(cosangle)
        # reflex angle
        if np.cross(v1,v2) < 0 :
            angle = 2*np.pi - angle
        return angle

    # couldn't compute
    return None

OUT_OF_RANGE=1e10

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# UTILS ---------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Manages polynomial chains
class Polychain(object) :
    """
    Manages polynomial chains.
    Given a polychain it can compute

    autocollision           collisions
    get_point(distance)     coordinates of points in the 2D space length
    isPointInChain          a point belonging or not to the chain

    """

    # versor defining the absolute reference frame for angles
    HVERSOR = np.array([1,0])

    def __init__(self):

        # store variables for graphic debug
        self.collision_data = np.ones([6,2])*OUT_OF_RANGE

    def set_chain(self, chain) :
        '''
         Configure the polychain with a given list of points

         :param     chain       the polygonal chain
         :type      chain       a list of (x,y) points
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

    q + s = sq     p + r = rp
            \       /
             \     /
              \   /
               \ /
      p + t*r   \ q + u*s
               / \
              /   \
             /     \
            p       q


        :param  epsilon             a trheshold of sensitivity of
                                    each point of the chain (how much they
                                    must be close one another to "touch")

        :param  is_set_collinear    further test for collinearity (segments are
                                    parallel and non-intersecting)

        :param  debug               if collision_data has to be stored

        :return  True in case of collision
        '''

        self.intersect = None
        (start, end) = self.chain[[1,-1]]

        n = len(self.chain)
        rng = range(1,n)

        # store variables for graphic debug
        if debug == True:
            self.collision_data = np.ones([6,2])*OUT_OF_RANGE

        # iterate over all combinations of pairs of segments
        # of the polychain and for each pair compute intersection
        for x in rng :

            # points of the first segment
            p = self.chain[x-1]    # start point
            rp = self.chain[x]    # end point
            r = rp - p    # end point relative to the start point

            for y in rng :

                # points of the seco_-nd segment
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
                                    if debug == True:
                                        self.collision_data[0] = p
                                        self.collision_data[1] = rp
                                        self.collision_data[3] = q
                                        self.collision_data[4] = sq
                                    return True


                    # segments are not parallel
                    if not rxs_zero :
                        t = qpxs / rxs
                        u = qpxr / rxs

                        int1 = p+t*r
                        int2 = q+u*s

                        # segments intersect
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            self.intersect = int1
                            if debug == True:
                                self.collision_data[0] = p
                                self.collision_data[1] = rp
                                self.collision_data[2] = int1
                                self.collision_data[3] = q
                                self.collision_data[4] = sq
                                self.collision_data[5] = int2
                            return True
        # no intersection
        return False

    def manage_auto_collision(self, curr_chain, move_back_fun, **kargs):

        self.set_chain(curr_chain)

        count_collisions = 1
        c_scale = 20.0
        delta = np.pi*(1.0/15.0)

        is_colliding = self.autocollision(**kargs)
        collided = is_colliding

        while is_colliding == True :
            if count_collisions <  c_scale :

                curr_chain = move_back_fun(
                    count_collisions = count_collisions,
                    delta = delta,
                    c_scale = c_scale)

                self.set_chain(curr_chain)

                is_colliding = self.autocollision(**kargs)

                count_collisions += 1

        return collided, curr_chain

    def isPointInChain(self, point, epsilon = 0.1 ) :
        '''
            find out if a point belongs to the chain.

            :param  point       a 2D point
            :param  epsilon     a minimal distance to discriminate
                                belonging points from non-belongingg ones.

            :return             a list of distances correspoding to the the line
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

        :param  distance    the proportional distance within the polychain.
                            can be a number between 0 and 1.
        :type   distance    float

        :return             the 2D coordinates of the point

        '''

        if distance > 1 :
            raise ValueError('distance must be a proportion of the polyline length (0,1)')

        if distance == 0.0 :
            return self.chain[0]

        if distance == 1.0 :
            return self.chain[-1]

        scaled_distance = sum(self.segment_lengths)*distance
        cumulated_length = 0

        # find the length until the last segment before
        # the one on which the point is
        for segment_index in xrange(self.vertices_number-1) :
            current_segment_length = self.segment_lengths[segment_index]
            if cumulated_length <= scaled_distance <= \
                    (cumulated_length + current_segment_length) :
                break
            cumulated_length += self.segment_lengths[segment_index]

        # evaluate the the distance from the last segment before
        # the one on which the point is
        last_segment_distance = scaled_distance - cumulated_length

        # evaluate coordinates relative to the last edge before the point
        last_segment_x = last_segment_distance*np.cos( \
                sum(self.segment_angles[:(segment_index+1)]) )
        last_segment_y = last_segment_distance*np.sin( \
                sum(self.segment_angles[:(segment_index+1)]) )

        # return the absolute coordinates
        return self.chain[segment_index] + (last_segment_x, last_segment_y)


    def get_dense_chain(self, density) :
        """
        add points to the polyvhain so to increase the number of segments
        """

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
# ARM KINEMATICS ------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

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
        :param          number_of_joint    number of joints
        :param          joint_angles       initial joint angles
        :param          joint_lims         joint angles limits
        :param          segment_lengths    length of arm segmens
        :param          origin             origin coords of arm
        :param          mirror             if the angles are all mirrowed

        :type           number_of_joint    int
        :type           joint_angles       list
        :type           joint_lims         list
        :type           segment_lengths    list
        :type           origin             list
        :type           mirror             bool

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
        :param      joint_angles    current angles of the joints
        :type       joint_angles    array(n)

        :return                    'number of joint' [x,y] coordinates and angles
        :rtype                      tuple(array(n,2), array(n))

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

def oscillator(x, scale, freqs):
    '''
    :param  x       list of timesteps
    :param  scale   scaling factor for the frequency
    :param  freqs   list of frequencies (one for each trjectory)
    '''

    x = np.array(x)  # timeseries
    trajectories_num = freqs.size
    freqs = np.array(freqs)
    x = np.outer(x, np.ones(trajectories_num))

    return (np.cos(2.0 * np.pi * freqs * x / scale + np.pi) + 1.0) * 0.5

#----------------------------------------------------------------------

def test_collision(plot=True):

    if plot == True:
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

    langle =  np.vstack([ np.pi*oscillator(xline,15,np.random.rand(3)) for i in range(trials) ])
    rangle =  np.vstack([ np.pi*oscillator(xline,15,np.random.rand(3)) for i in range(trials) ])
    lpos,langle[0,:] = larm.get_joint_positions(langle[0,:])
    rpos,rangle[0,:] = rarm.get_joint_positions(rangle[0,:])
    rbt = np.vstack((lpos[::-1],rpos))

    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect="equal")
        robot_img, = plt.plot(*rbt.T)
        robot_jts = plt.scatter(*rbt.T)
        prpl, = plt.plot(*(np.ones([2,2])*OUT_OF_RANGE).T, lw=3, c="red")
        prpp = plt.scatter(*(np.ones([2,2])*OUT_OF_RANGE).T, s=60, c="red")
        qsql, = plt.plot(*(np.ones([2,2])*OUT_OF_RANGE).T, lw=3, c="green")
        qsqp = plt.scatter(*(np.ones([2,2])*OUT_OF_RANGE).T, s=60, c="green")
        prtp = plt.scatter(*(np.ones([2,2])*OUT_OF_RANGE).T, s=120, color="yellow")
        plt.xlim([-5,5])
        plt.ylim([-0.5,3.0])

    def move(x) :
        lpos,langle[x,:] = larm.get_joint_positions(langle[x,:])
        rpos,rangle[x,:] = rarm.get_joint_positions(rangle[x,:])
        return np.vstack((lpos[::-1],rpos))

    class MoveBackFunct:
        def __init__(self, x, langle, rangle):
            self.x = x
            self.langle = langle.copy()
            self.rangle = rangle.copy()

        def __call__(self, count_collisions = 1,
                      delta = np.pi*(1.0/15.0),
                      c_scale = 20.0) :

            x = self.x
            delta_langle = ( (self.langle[x,:] - self.langle[x-1,:])
                            if not x<0 else self.langle[x,:]*0.0 )

            curr_langle = (self.langle[x,:]
                       - count_collisions
                       * delta*(count_collisions**0.5)
                       * np.sign(delta_langle)
                       / float(c_scale*2) )

            delta_rangle = ( (self.rangle[x,:] - self.rangle[x-1,:])
                            if not x<0 else self.rangle[x,:]*0.1 )
            curr_rangle = (self.rangle[x,:]
                       - count_collisions
                       * delta*(count_collisions**0.5)
                       * np.sign(delta_rangle)
                       / float(c_scale*2) )

            lpos,langle = larm.get_joint_positions(curr_langle)
            rpos,rangle = rarm.get_joint_positions(curr_rangle)

            return np.vstack((lpos[::-1],rpos))

    for x in range(stime*trials):

        collided, curr_chain = polychain.manage_auto_collision(
            curr_chain =  move(x),
            move_back_fun = MoveBackFunct(x, langle, rangle),
            epsilon = 0.1,
            is_set_collinear=True,
            debug=True)

        (p,rp,int1,q,sq,int2) = polychain.collision_data

        if plot == True:
            robot_img.set_data(curr_chain.T)
            robot_jts.set_offsets(curr_chain)

            if collided == True:
                prpp.set_offsets( np.vstack([p,rp]) )
                qsqp.set_offsets( np.vstack([q,sq]) )
                prtp.set_offsets( np.vstack([int1,int2]) )
                prpl.set_data( *np.vstack([p,rp]).T )
                qsql.set_data( *np.vstack([q,sq]).T )
            else:
                prpp.set_offsets( np.ones([2,2])*OUT_OF_RANGE )
                qsqp.set_offsets( np.ones([2,2])*OUT_OF_RANGE )
                prtp.set_offsets( np.ones([2,2])*OUT_OF_RANGE )
                prpl.set_data( *(np.ones([2,2])*OUT_OF_RANGE).T )
                qsql.set_data( *(np.ones([2,2])*OUT_OF_RANGE).T )

            fig.canvas.draw()
            plt.pause(0.05)
        else:
            print x,if_coll

    raw_input()


if __name__ == "__main__" :

    test_collision()
