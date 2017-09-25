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

import copy
import math
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
    """ Calculate the cross product between two vectors in a
        2D space (redefined for speed)

    :param a: first vector
    :type a: numpy.array(n, dtype=float)

    :param b: second vector
    :type b: numpy.array(n, dtype=float)

    """

    return a[0] * b[1] - a[1] * b[0]

def norm(a):
    """ Calculate the norm of a vecto in a
        2D space (redefined for speed)

    :param a: the vector
    :type a: numpy.array(n, dtype=float)

    """

    return math.sqrt(a[0]**2 + a[1]**2)





def project(a, b, c):
    """ Projects the point c into the segment ab
    
        :param a: the start-point of th segment
        :type a: any container with a pair of doubles
        
        :param b: the end-point of th segment
        :type b: any container with a pair of doubles
        
        :param c: the point to project
        :type c: any container with a pair of doubles

        :return: the projected point or None if cannot be projected
        :rtype: numpy.array(2) or None

    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    d = a + np.dot(c-a,b-a)/ np.dot(b-a,b-a) * (b-a)
    
    # if sum of the distances of d form a and b is equal to 
    # the distance of a from b then d belongs to the segment ab
    between = ((norm(a - d) + 
                norm(b - d) ) / 
               norm(b - a))
    
    if between == 1 :
        return d
    else:
        return None



def get_angle(v1, v2):
    """ Calculate the angle between two vectors in a 2D space

    see https://goo.gl/Uq0lw

    :param v1: first 2D edge
    :type v1: numpy.array(2, dtype=float)

    :param v2: second 2D edge
    :type v2: numpy.array(2, dtype=float)

    :return: angle (radiants)
    :rtype: float
    """

    # must be both segments (not points)
    nv1 = norm(v1)
    nv2 = norm(v2)
    nv1v2 = nv1*nv2
    if (nv1v2) != 0:
        cosangle = np.dot(v1, v2) / nv1v2
        cosangle = np.maximum(-1, np.minimum(1, cosangle))
        angle = np.arccos(cosangle)
        # reflex angle
        if np.cross(v1, v2) < 0:
            angle = 2 * np.pi - angle
        return angle

    # couldn't compute
    return None


OUT_OF_RANGE = 1e10

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# COLLISIONS ----------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Manages polynomial chains


class Polychain(object):
    """ Manages polynomial chains.

    Given a polychain it can compute:

    get_point(distance) -> coordinates of points in the 2D space length
    isPointInChain -> a point belonging or not to the chain

    """

    # versor defining the absolute reference frame for angles
    HVERSOR = np.array([1, 0])

    def __init__(self):

        # store variables for graphic debug
        self.collision_data = np.ones([6, 2]) * OUT_OF_RANGE

    def set_chain(self, chain):
        ''' Configure the polychain with a given list of points

        :param chain: the polygonal chain
        :type chain: list(numpy.array((n,2), dtype=float))
        '''

        # ensure it is a numpy array
        self.chain = np.array(chain[:])
        # the length of the chain (number of vertices)
        self.vertices_number = len(self.chain)
        # the chain must be greater than one point
        if self.vertices_number < 2:
            raise ValueError('Polychain initialized with only one point. ' +
                             'Minimum required is two.')

        # calculate segments lengths
        self.segment_lengths = [
            norm(self.chain[x] - self.chain[x - 1])
            for x in xrange(1, self.vertices_number)]

        # calculate angles at the vertices
        self.segment_angles = []
        for x in xrange(1, self.vertices_number):
            # define ab segment. (the first ab segment is the versor defining
            #   horizontal axis)
            ab = self.HVERSOR if x == 1 else self.chain[x -
                                                        1] - self.chain[x - 2]
            # define bc segment
            bc = self.chain[x] - self.chain[x - 1]
            # compute abVbc angle and add it to segment_angles
            self.segment_angles.append(get_angle(ab, bc))

    def isPointInChain(self, point, epsilon=0.1):
        ''' Find out if a point belongs to the chain.

        :param point: a 2D point
        :type point: numpy.ndarray(2, dtype=float)

        :param epsilon: a minimal distance to discriminate
                            belonging points from non-belongingg ones.
        :type epsilon: float

        :return: a list of distances correspoding to the the line
                    intersection with that point. Empty list if
                    the point does not belong to the chain
        :rtype: list(float)
        '''

        distances = []
        c = np.array(point)
        for x in xrange(1, len(self.chain)):

            a = self.chain[x - 1]
            b = self.chain[x]

            # check if the  point is within the same line
            if np.any(c != a) and np.any(c != b):
                if math.fabs(cross(b - a, c - a)) < epsilon:

                    projected_point = project(a,b,c)

                    if projected_point is not None:
                        
                        distance = np.sum(self.segment_lengths[:(x - 1)])
                        distance += norm(projected_point - self.chain[x - 1])
                        distance = distance / sum(self.segment_lengths)
                        distances.append(distance)
                        
            elif np.all(c == a):
        
                distance = np.sum(self.segment_lengths[:(x - 1)])
                distance = distance / sum(self.segment_lengths)
                distances.append(distance)

            elif np.all(c == b):
                
                distance = np.sum(self.segment_lengths[:x])
                distance = distance / sum(self.segment_lengths)
                distances.append(distance)

        return list(set(distances))

    def get_point(self, distance):
        ''' Get a point in the 2D space given a
            distance from the first point of the chain

        :param distance: the proportional distance within the polychain.
                            can be a number between 0 and 1.
        :type distance: float

        :return: the 2D coordinates of the point
        :rtype: numpy.array(2, dtype=float)

        '''

        if distance > 1:
            raise ValueError('distance must be a proportion of the polyline' +
                             'length (0,1)')

        if distance == 0.0:
            return self.chain[0]

        if distance == 1.0:
            return self.chain[-1]

        scaled_distance = sum(self.segment_lengths) * distance
        cumulated_length = 0

        # find the length until the last segment before
        # the one on which the point is
        for segment_index in xrange(self.vertices_number - 1):
            current_segment_length = self.segment_lengths[segment_index]
            if cumulated_length <= scaled_distance <= \
                    (cumulated_length + current_segment_length):
                break
            cumulated_length += self.segment_lengths[segment_index]

        # evaluate the the distance from the last segment before
        # the one on which the point is
        last_segment_distance = scaled_distance - cumulated_length

        # evaluate coordinates relative to the last edge before the point
        last_segment_x = last_segment_distance * np.cos(
            sum(self.segment_angles[:(segment_index + 1)]))
        last_segment_y = last_segment_distance * np.sin(
            sum(self.segment_angles[:(segment_index + 1)]))

        # return the absolute coordinates
        return self.chain[segment_index] + (last_segment_x, last_segment_y)

    def get_dense_chain(self, density):
        """ Add points to the polyvhain so to increase the number of segments

        :param density: the number of points of the reformatted chain
        :type density: int

        :return: the new chain
        :rtype: numpy.ndarray((density, 2), dtype=float)
        """

        dense_chain = []
        segment_number = density - 1

        dense_chain.append(self.get_point(0))

        for x in xrange(segment_number):
            dense_chain.append(self.get_point((1 + x) / float(segment_number)))

        return np.vstack(dense_chain)

    def get_length(self):
        '''
        :return: the length of the current polyline
        :rtype: int
        '''
        return sum(self.segment_lengths)

    def get_angles(self):

        angles = np.zeros(self.vertices_number - 1)

        for i, point in enumerate(self.chain[:-1]):
            p0 = self.chain[0] - self.HVERSOR if i < 1 else self.chain[i - 1]
            p1 = point
            p2 = self.chain[i + 1]
            angle1 = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            angle2 = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
            angles[i] = np.pi - (angle1 - angle2)

        return (angles > np.pi) * np.pi - angles
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    # COLLISIONS ----------------------------------------------------------
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------


class Collision(object):
    ''' Detect if one or more chains are colliding

    Reads intersections between all the segments of the chains iterating over
    all combinations of segments in the chains

    Intersection algorithm from http://stackoverflow.com/a/565282/4209308

    for each pair of segments:
        first segment:
            p is one endpoint
            r is the coordinates relative to the first endpoint
                of the second endpoint
            t is a scalar defining the distance of the intersection point
                from p on the first segment
        second segment:
            q is one endpoint
            s is the coordinates relative to the first endpoint
                of the second endpoint
            u is a scalar defining the distance of the intersection point
                from p on the first segment

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

    '''

    def __init__(self, debug=False):
        if debug == True:
            # initialize info structure for the collision
            self.collision_data = [
                np.ones(2) * OUT_OF_RANGE,  # p
                np.ones(2) * OUT_OF_RANGE,  # rp
                np.ones(2) * OUT_OF_RANGE,  # p + t*r
                np.ones(2) * OUT_OF_RANGE,  # q
                np.ones(2) * OUT_OF_RANGE,  # sq
                np.ones(2) * OUT_OF_RANGE]  # q + u*s

    def __call__(self, chains, epsilon=0.1, is_set_collinear=True, debug=False):
        ''' all operator that computes collisions

            :param chains: a list of chains (lists of points)
            :type chains: list(numpy.ndarray(n, 2))

            :param epsilon: a trheshold of sensitivity of
                            each point of the chain (how much they
                            must be close one another to "touch")
            :type epsilon: float

            :param is_set_collinear: further test for collinearity (segments are
                                        parallel and non-intersecting)
            :type is_set_collinear: bool

            :param debug: if collision_data has to be stored
            :type debug: bool

            :return: True in case of collision
            :rtype: bool
        '''

        # initialize intersection flag
        self.intersect = None

        # ----- control the integrity of the chains argument ------

        # precondition : the argument must be an array or
        #   a list of arrays
        single_chain = type(chains) == np.ndarray
        multiple_chains = (type(chains) == list and
                           np.all([type(chain) == np.ndarray
                                   for chain in chains]))
        assert(multiple_chains or single_chain)

        # make a unique chain - precondition (all objects within
        #   "chains" muts have the same length on their last dimension)
        self.chain = np.vstack(chains)

        # precondition: elements must be points
        assert(self.chain.shape[-1] == 2)

        # ---------------------------------------------------------

        # get chain's info
        (start, end) = self.chain[[1, -1]]
        n = len(self.chain)
        rng = range(1, n)

        # if more than one chain is given
        if multiple_chains:
            # compute the lengths of the given chains
            chain_lens = [len(chain) for chain in chains]
            # throw away the indices corresponfding to the begin
            #   of each original chain
            rng = list(set(rng) - set(np.cumsum(chain_lens)))

        # store variables for graphic debug
        if debug == True:
            self.collision_data = np.ones([6, 2]) * OUT_OF_RANGE

        # iterate over all combinations of pairs of segments
        # of the polychain and for each pair compute intersection
        for x in rng:

            # points of the first segment
            p = self.chain[x - 1]    # start point
            rp = self.chain[x]    # end point
            r = rp - p    # end point relative to the start point

            for y in rng:

                # points of the second segment
                q = self.chain[y - 1]    # start point
                sq = self.chain[y]    # end point
                s = sq - q    # end point relative to the start point

                # test if start or end points overlap
                junction = np.all(p == sq) or np.all(q == rp)

                # only compute collisions if end points are not
                # junktions
                if x != y and not junction:

                    # cross product of the relative end points.
                    # if rxs = 0 the two segments are parallels
                    rxs = cross(r, s)
                    rxs_zero = abs(rxs) < epsilon

                    qpxr = cross(q - p, r)
                    qpxs = cross(q - p, s)

                    # segments are parallel
                    if rxs_zero:
                        if is_set_collinear:

                            # segments are collinear
                            if abs(qpxr) < epsilon:

                                t0 = np.dot(q - p, r / np.dot(r, r))
                                t1 = t0 + np.dot(s, r / np.dot(r, r))

                                mint = max(t0, 0)
                                maxt = min(t1, 1)

                                # segments overlap
                                if mint < maxt:
                                    if debug == True:
                                        self.collision_data[0] = p
                                        self.collision_data[1] = rp
                                        self.collision_data[3] = q
                                        self.collision_data[4] = sq
                                    return True

                    # segments are not parallel
                    if not rxs_zero:
                        t = qpxs / rxs
                        u = qpxr / rxs

                        int1 = p + t * r
                        int2 = q + u * s

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


class CollisionManager(object):
    """ Manages collisions

    Uses a Collision object to detect collisions and
    calls a backward-move function object in case of
    compenetrations
    """

    def __init__(self):

        self.collision = Collision()
        self.chain = Polychain()

    def manage_collisions(self, prev_chain, curr_chain, move_back_fun,
                          other_chains=None, delta=1.0, substeps=50.0,
                          min_distance=np.pi / 16.0, do_substep=True,
                           **kargs):
        """ Manage current collisions.

        If the current set of positions (curr_chain)
        has collisions produces substeps with beckward-moving positions until
        the chain does not collide anymore.

        :param prev_chain: previous position (at timestep t - 1)
        :type prev_chain: numpy.ndarray((n, 2), dtype=float)

        :param curr_chain: current position (at timestep t)
        :type curr_chain: numpy.ndarray((n, 2), dtype=float)
        
        :param move_back_fun: a function object generating backward moves
        :type move_back_fun: function(count_substeps, delta, substeps):

                :param count_substeps: current substep number
                :type count_substeps: int

                :param delta: scaling factor of a beckward move
                :type delta: float

                :param substeps: the number of maximum substeps
                :type substeps: int
        
                
        :param other_chains: chains representing other objects that can collide with curr_chain
        :param  other_chains: list(numpy.ndarray((n, 2), dtype=float), ...)

        :param delta: scaling factor of a beckward move
        :type delta: float

        :param substeps: the number of maximum substeps
        :type substeps: int

        :param min_distance: the threshold distance over which we
                             compute substeps
        :type min_distance: float
        
        :param do_substep: do substeps (recursively) if distance is too much
        :type do_substep: bool
        
        :param **kargs: arguments for autocollision

        :return: a tuple with the test for collision and the current
                    position after collision
        :rtype: tuple(bool, numpy.ndarray((n, 2), dtype=float))

        """
        
        curr_chain = curr_chain.copy()
        prev_chain = prev_chain.copy()
        if other_chains is not None:
            other_chains = other_chains[:]
        
        # coumpute distance between prev and curr positions
        distance = np.linalg.norm(prev_chain - curr_chain)

        # init the counter of substeps
        count_substeps = 1

        # set the current positions as
        # the current polychain
        self.chain.set_chain(curr_chain)

        # detect collisions
        coll_chain = curr_chain.copy()
        if other_chains is not None:
            coll_chain = []
            coll_chain.append(curr_chain)
            for obj in other_chains:
                coll_chain.append(obj)

        is_colliding = self.collision(coll_chain, **kargs)
        collided = is_colliding

        # if there is a collision loop through substeps
        while is_colliding == True or distance > min_distance:
            if count_substeps < substeps:

                # do a backward move
                self.chain.set_chain(curr_chain)
                curr_chain = move_back_fun(
                    count_substeps=count_substeps,
                    delta=delta,
                    substeps=substeps)

                # set the current positions as
                # the current polychain
                self.chain.set_chain(curr_chain)

                # detect collisions
                coll_chain = curr_chain.copy()
                if other_chains is not None:
                    coll_chain = []
                    coll_chain.append(curr_chain)
                    for obj in other_chains:
                        coll_chain.append(obj)

                is_colliding = self.collision(coll_chain, **kargs)

                # update substep counter
                count_substeps += 1
            else:
                # too many substeps, step back to
                # the previous (non colliding) position
                return collided, prev_chain.copy()

        return collided, curr_chain.copy()

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ARM KINEMATICS ------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class Arm(object):
    """ Kinematics of a number_of_joint-degrees-of-freedom 2-dimensional arm.

    Given the increment of joint angles calculate
    the current positions of the edges of the
    arm segments.

    """

    def __init__(self,
                 number_of_joint=3,
                 joint_lims=None,
                 segment_lengths=None,
                 origin=[0, 0],
                 mirror=False
                 ):
        """
        :param number_of_joint: number of joints
        :type number_of_joint: int

        :param joint_angles: initial joint angles
        :type joint_angles: numpy.ndarray(number_of_joint, dtype=float)

        :param joint_lims: joint angles limits
        :type joint_lims: numpy.ndarray((number_of_joint, 2), dtype=float)

        :param segment_lengths: length of arm segments
        :type param segment_lengths: numpy.ndarray(number_of_joint, dtype=int)

        :param origin: origin coords of arm
        :type origin: numpy.ndarray(2, dtype=float)

        :param mirror: if the angles are all mirrowed
        :type mirror: bool

        """

        self.mirror = mirror
        self.number_of_joint = number_of_joint

        # initialize lengths
        if segment_lengths is None:
            segment_lengths = np.ones(number_of_joint)
        self.segment_lengths = np.array(segment_lengths)

        # initialize limits
        if joint_lims is None:
            joint_lims = vstack([-np.ones(number_of_joint) *
                                 np.pi, ones(number_of_joint) * np.pi]).T
        self.joint_lims = np.array(joint_lims)

        # set origin coords
        self.origin = np.array(origin)

    def get_joint_positions(self,  joint_angles):
        """ Finds the (x, y) coordinates of each joint

        :param joint_angles: current angles of the joints
        :type joint_angles: numpy.ndarray(n)

        :return: 'number of joint' [x,y] coordinates and angles
        :rtype: tuple(
                    numpy.ndarray((n, 2), dtype=float),
                    numpy.ndarray(n, dtype=float) )
        """

        # current angles
        res_joint_angles = joint_angles.copy()

        # detect limits
        maskminus = res_joint_angles > self.joint_lims[:, 0]
        maskplus = res_joint_angles < self.joint_lims[:, 1]

        res_joint_angles = res_joint_angles * (maskplus * maskminus)
        res_joint_angles += self.joint_lims[:, 0] * (np.logical_not(maskminus))
        res_joint_angles += self.joint_lims[:, 1] * (np.logical_not(maskplus))

        # mirror
        if self.mirror:
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
                np.cos((res_joint_angles[:(k + 1)]).sum())
                for k in range(j + 1)
            ])
            for j in range(self.number_of_joint)
        ])

        # translate to the x origin
        x = np.hstack([self.origin[0], x + self.origin[0]])

        # calculate y coords of arm edges.
        # the y-axis position of each edge is the
        # sum of its projection on the x-axis
        # and all the projections of the
        # previous edges
        y = np.array([
            sum([
                self.segment_lengths[k] *
                np.sin((res_joint_angles[:(k + 1)]).sum())
                for k in range(j + 1)
            ])
            for j in range(self.number_of_joint)
        ])

        # translate to the y origin
        y = np.hstack([self.origin[1], y + self.origin[1]])

        pos = np.array([x, y]).T

        # de-mirror
        if self.mirror:
            res_joint_angles[0] -= np.pi
            res_joint_angles = -res_joint_angles

        return (pos, res_joint_angles)
    
    def get_joint_angles(self, joint_positions):
        ''' Get angles from a set of positions
            
            :param joint_positions: the positions 
                       from which we compute angles
            :type joint_positions: numpy.array(number_of_joint,
                        dtype=float)
        '''
        
        chain = Polychain()
        chain.set_chain(joint_positions)
        
        return chain.get_angles()
        
        

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# TEST ----------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class Oscillator(object):
    def __init__(self, scale, freqs):
        '''
        :param scale: scaling factor for the frequency
        :type scale: float

        :param freqs: list of frequencies (one for each trjectory)
        :type freqs: list(float)
        '''

        self.scale = scale
        self.freqs = np.array(freqs)
        self.trajectories_num = freqs.size
        self.t = 0

    def __call__(self):
        """ Calls an oscillator's step

        :return: the current angles after one step
        :rtype: numpy.array(trajectories_num, dtype=float)s
        """
        res = -np.pi * \
            (np.sin(((2.0 * np.pi * self.freqs * (self.t / float(self.scale))) + np.pi))) * 0.5
        self.t += 1

        return res

    def reset(self):

        self.t = 0


def test_collision():
    """
    Test simulation for autocollisions
    """

    collision_manager = CollisionManager()

    object_pos = np.array([[-1.0,  -1.0], [1.0,  -1.0], [1.0,   1.0],
                           [-1.0,   1.0], [-1.0,  -1.0]])
    object_pos = object_pos * 0.4 + [-1, 1]

    # object that manages the left arm (mirrowed)
    arm = Arm(
        number_of_joint=3,  # 3 joints
        origin=[-1.5, 0.0],  # origin at (1.0 , 0.5)
        segment_lengths=np.array([1, 1, 1]),
        joint_lims=[
            [0, np.pi * 0.9],    # first joint limits
            [0, np.pi * 0.6],    # second joint limits
            [0, np.pi * 0.6],    # third joint limits
        ],
        mirror=True
    )

    def move(angle):
        """ move function - simply browse the joint timeseries

        :param angle: the commanded angle
        :type angle: float

        :return: the positions and angles after move
        :rtype: tuple(
                    numpy.ndarray((n, 2), type=float),
                    numpy.ndarray(n, type=float))
        """

        return arm.get_joint_positions(angle)

    class MoveBackFunct:
        """ Function object for backward moves
        """

        def __init__(self, init_angle, curr_angle):
            """
            Share data with the main function

            :param inital_angle: the initial angle before the step
            :type inital_angle: numpy.ndarray(n, type=float))

            :param curr_angle: the current angle (at the end of the step)
            :type curr_angle: numpy.ndarray(n, type=float))
            """
            self.inital_angle = init_angle.copy()
            self.curr_angle = curr_angle.copy()
            self.arm = copy.copy(arm)

        def __call__(self, count_substeps, delta, substeps):
            """ Computes a backward step

            :param count_substeps: current substep number
            :type count_substeps: int

            :param delta: scaling factor of a beckward move
            :type delta: float

            :param substeps: the number of maximum substeps
            :type substeps: int

            """
            # the gap between the current and the previous arm joint angles
            delta_angle = (self.curr_angle - self.inital_angle)

            # compute the substep angle
            curr_angle = (self.curr_angle
                          - (count_substeps / float(substeps))
                          * delta * np.sign(delta_angle))

            # compute positions
            pos, angle = self.arm.get_joint_positions(curr_angle)

            # return the arm positions
            return pos

    # animate
    stime = 30
    scale = 5.0
    freqs = np.random.rand(3)

    curr_angle = np.zeros(3)
    curr_arm_pos, curr_angle = move(curr_angle)

    # prepare graphic objects
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    plt.xlim([-5, 5])
    plt.ylim([-0.5, 3.0])

    # inital plots
    object_img, = plt.plot(*object_pos.T)
    robot_img, = plt.plot(*curr_arm_pos.T)
    robot_jts = plt.scatter(*curr_arm_pos.T)

    cm = Polychain()

    collided = False
    for k in xrange(10):
        freqs = np.random.rand(3)
        oscillator = Oscillator(scale, freqs)
        curr_angle = oscillator()
        curr_angle = np.zeros(3)
        curr_arm_pos, curr_angle = move(curr_angle)

        if k == 2:
            object_pos += 1.0
            object_img.set_data(*object_pos.T)
        for t in xrange(stime):
            print t
            prev_angle = curr_angle.copy()
            prev_arm_pos = curr_arm_pos.copy()

            curr_angle += oscillator()
            curr_arm_pos, curr_angle = move(curr_angle)

            # manages current move
            collided, curr_arm_pos = collision_manager.manage_collisions(
                prev_chain=prev_arm_pos,
                curr_chain=curr_arm_pos,
                other_chains=object_pos,
                move_back_fun=MoveBackFunct(
                    init_angle=prev_angle, curr_angle=curr_angle),
                epsilon=0.1)

            if collided:
                cm.set_chain(curr_arm_pos)
                curr_angle = cm.get_angles()

            # plot the body
            robot_img.set_data(*curr_arm_pos.T)
            robot_jts.set_offsets(curr_arm_pos)

            fig.canvas.draw()
            plt.pause(.5)


if __name__ == "__main__":

    import matplotlib
    matplotlib.use("QT4Agg")
    import matplotlib.pyplot as plt
    
    plt.ion()
    #test_collision()
    
    cm = Polychain()
    cm.set_chain([[-2,0],
                  [-1,0],
                  [0,0],
                  [1,0],
                  [1,1] ])
    
    print cm.isPointInChain([.95,0.05], 0.1)
    print cm.isPointInChain([1.,0.0], 0.1)
    print cm.isPointInChain([2.,0.0], 0.1)
