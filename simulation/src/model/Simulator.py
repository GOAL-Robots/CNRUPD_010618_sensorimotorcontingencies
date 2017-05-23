#!/usr/bin/env python
from __future__ import division
import sys
import numpy as np
import time

from model import *
from gauss_utils import TwoDimensionalGaussianMaker as GM
import kinematics as KM

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


class PerceptionManager(object):
    '''
    Manages the computation of sensory inputs
    '''

    def __init__(self, pixels, lims, touch_th,
                 num_touch_sensors, touch_sigma, collision=KM.Collision()):
        '''
        :param  pixels                  width and length of the retina
        :param  lims                    x and y limits of the field of view
        :param  touch_th                threshold of sensor activation to detect touch
        :param  num_touch_sensors       number of sensors in the body
        :param  touch_sigma             standard deviation of the sensor radial bases
        :type   pixels                  list - [width, height]
        :type   lims                    list of pairs   - [[xmin, xmax], [ymin, ymax]]
        :type   touch_th                float
        :type   num_touch_sensors       int
        :type   touch_sigma             float

        '''

        self.pixels = np.array(pixels)
        self.lims = np.vstack(lims)
        self.touch_th = touch_th
        self.num_touch_sensors = num_touch_sensors
        self.touch_sigma = touch_sigma
        self.collision = collision

        # get the width and height of the visual field
        self.size = (self.lims[:, 1] - self.lims[:, 0])
        # get the vertical middle of the visual field
        self.ycenter = self.lims[1, 0] + self.size[1] / 2
        # init the chain object that computes the current positions of sensors
        self.chain = KM.Polychain()

        # compute the witdh and height magnitudes of the gaussians in
        # proprioception retina
        self.proprioceptive_retina_sigma = (self.size *
                                            pm_proprioceptive_retina_sigma)
        # compute the witdh and height magnitudes of the gaussians in touch
        # retina
        self.touch_retina_sigma = self.size * pm_touch_retina_sigma

        # scale width of the gaussians in proprioception retina
        self.proprioceptive_retina_sigma[0] *= pm_proprioceptive_retina_sigma_width_scale
        # scale width of the gaussians in touch retina
        self.touch_retina_sigma[0] *= pm_touch_retina_sigma_width_scale

        # init the gaussian maker to build the 2d gaussians for the retinas
        self.gm = GM(lims=np.hstack([self.lims, self.pixels.reshape(2, 1)]))

        # the resolution of each body part in the visual retina (number of
        # points)
        self.image_resolution = pm_image_resolution - 2

        # initially set the body chain to a 2-points line
        self.chain.set_chain(np.vstack(([-1, 0], [1, 0])))

        # compute the initial sensors' positions
        self.sensors = self.chain.get_dense_chain(self.num_touch_sensors)

        self.sensors_prev = self.chain.get_dense_chain(self.num_touch_sensors)

    def get_image(self, body_tokens):
        '''
        Builds the retina image from the current positions of the body

        :param  body_tokens     list of body parts. Each part is a list of 2D points
        :type   body_tokens     list of list of pairs - [ [[x0,y0], ..., [xn,yn]], ... ]

        :return a 2D array
        :rtype  np.array(self.size, dtype=float )

        '''

        # init image
        image = np.zeros(self.pixels)

        # for each token add all points to the 2D image
        for c in body_tokens:
            # compute the positions of the token's points
            self.chain.set_chain(c)
            # compute the position of the new poins given the current
            # resolution
            dense_chain = self.chain.get_dense_chain(self.image_resolution)

            # add each point to the image
            for point in dense_chain:
                # compute the discrete coordinates of the point in terms of the
                # retina
                p = np.floor(
                    ((point - self.lims[:, 0]) / self.size) * self.pixels).astype("int")
                # add points only if within the retina margins
                if np.all(p < self.pixels) and np.all(p > 0):
                    image[p[0], p[1]] += 1

        # ensure that max value is at most 1
        image /= image.max()

        # TODO: control all rotations  (it works, but why we transpose this?)
        return image.T

    def get_proprioception(self, angles_tokens):
        '''
        Build the current proprioception retina.
        All body joints are viewed as points in a 1-dimensional space.
        At each joint position a 2D gaussian is computed with standard deviation
        on the vertical axis defined by the amplitude of the
        joint angle times self.proprioceptive_retina_sigma[1] and
        standard deviation on the orizontal axis defined by the amplitude of
        the joint angle times self.proprioceptive_retina_sigma[0].
        All gaussians are summed point-to-point to obtain a mixture.

        :param  angle_tokens    list of lists of angles of all body parts' joints
        :type   angle_token     listo f list of floats - [ [ja0, ..., jan], ... ]

        :return a 2D array
        :rtype  np.array(self.size, dtype=float )

        '''

        # init retina
        image = np.zeros(self.pixels).astype("float")

        # convrt the sequence of angle lists in a into a flatten vector
        all_angles = np.hstack(angles_tokens)

        # add a column with the number of joints to the lims array (x-axis) -  needed for
        # telling the number of bins to linspace
        lims = np.hstack([self.lims[0], len(all_angles)])

        # compute the center of joint gaussians within the retina
        abs_body_tokens = np.vstack([
            # x-positions are uniformly distributed within the retinal x-limits
            np.linspace(*lims),
            # y-positions are all alligned at the middel of the retinal
            # y-limits
            self.ycenter * np.ones(len(all_angles))
        ]).T

        # for each joint compute the gaussian
        for point, angle in zip(abs_body_tokens, all_angles):
            if abs(angle) > pm_proprioceptive_angle_threshold:
                # the standard deviation on the x and y axis depends on the
                # angle of the joint
                sigma = abs(angle) * self.proprioceptive_retina_sigma
                # build the 2d_gaussian on the retina-grid
                g = self.gm(point, sigma)[0]
                # add the current gaussian to the retina
                image += g.reshape(*self.pixels) * angle

        return image.T

    def get_touch(self, body_tokens):
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

        # compute touches

        # get the chain of the whole body (a single array of 2D points )
        bts = np.vstack(body_tokens)
        self.chain.set_chain(bts)

        # update sensors
        self.sensors_prev = self.sensors
        self.sensors = self.chain.get_dense_chain(self.num_touch_sensors)

        # init the vector of touch measures
        sensors_n = len(self.sensors)
        touches = np.zeros(sensors_n)

        # read contact of the two edges (hands) with the rest of the points
        for x, sensor in zip([0, sensors_n - 1], [self.sensors[0], self.sensors[-1]]):

            # for each edge iterate over the other points
            for y, point in enumerate(self.sensors):

                # do not count the edge with itself (it would be autotouch!!)
                if x != y and abs(y - x) > (self.num_touch_sensors / 4.0):

                    # touch is measured as a radial basis of the distance from
                    # the sensor
                    stouch = np.exp(-((np.linalg.norm(point - sensor))**2) /
                                    (2 * self.touch_sigma**2))

                    touches[y] += stouch

        # write information into the touch retina
        # init retina
        image = np.zeros(self.pixels)

        # add a column with the number of joints to the lims array (x-axis) -  needed to
        # tell the number of bins to linspace
        lims = np.hstack([self.lims[0], sensors_n])

        # compute the center of joint gaussians within the retina
        abs_body_tokens = np.vstack([
            # x-positions are uniformly distributed within the retinal x-limits
            np.linspace(*lims),
            # y-positions are all alligned at the middle of the retinal
            # y-limits
            self.ycenter * np.ones(sensors_n)
        ]).T

        # for each joint compute the gaussian
        for touch, point in zip(touches, abs_body_tokens):
            # the standard deviation on the x and y axis depends on the touch
            sigma = 1e-5 + touch * self.touch_retina_sigma
            # build the 2d_gaussian on the retina-grid
            g = self.gm(point, sigma)[0]
            # add the current gaussian to the retina if touch is greater than
            # threshold
            if touch > self.touch_th:
                image += g.reshape(*self.pixels)

        return image.T, touches

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


class KinematicActuator(object):
    '''
    Init and control the position of the two arms
    '''

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __init__(self):

        self.NUMBER_OF_JOINTS = ka_num_joints
        self.L_ORIGIN = ka_left_origin
        self.R_ORIGIN = ka_right_origin

        # build left arm (angle are mirrowed)
        self.arm_l = KM.Arm(
            number_of_joint=self.NUMBER_OF_JOINTS,    # 3 joints
            origin=self.L_ORIGIN,  # origin at (1.0 , 0.5)
            joint_lims=ka_left_lims,
            mirror=True
        )

        # build right arm
        self.arm_r = KM.Arm(
            number_of_joint=self.NUMBER_OF_JOINTS,    # 3 joints
            origin=self.R_ORIGIN,  # origin at (1.0 , 0.5)
            joint_lims=ka_right_lims
        )

        # init angles
        self.angles_l = np.zeros(self.NUMBER_OF_JOINTS)
        self.angles_r = np.zeros(self.NUMBER_OF_JOINTS)

        # get initial positions given initial angles
        self.position_l, _ = self.arm_l.get_joint_positions(self.angles_l)
        self.position_r, _ = self.arm_r.get_joint_positions(self.angles_r)

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def rescale_angles(self, l_angles, r_angles):

        l_mins = self.arm_l.joint_lims[:, 0]
        l_maxs = self.arm_l.joint_lims[:, 1]
        l_ranges = l_maxs - l_mins
        l_angles = l_angles * l_ranges + l_mins

        r_mins = self.arm_r.joint_lims[:, 0]
        r_maxs = self.arm_r.joint_lims[:, 1]
        r_ranges = r_maxs - r_mins
        r_angles = r_angles * r_ranges + r_mins

        return l_angles, r_angles

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def unscale_angles(self, l_angles, r_angles):

        l_mins = self.arm_l.joint_lims[:, 0]
        l_maxs = self.arm_l.joint_lims[:, 1]
        l_ranges = l_maxs - l_mins
        l_angles = (l_angles - l_mins) / l_ranges

        r_mins = self.arm_r.joint_lims[:, 0]
        r_maxs = self.arm_r.joint_lims[:, 1]
        r_ranges = r_maxs - r_mins
        r_angles = (r_angles - r_mins) / r_ranges

        return l_angles, r_angles

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def set_angles(self, angles_l, angles_r):
        '''
        Set angles and update positions

        :param  angles_l    angles of the joints of the left arm
        :param  angles_r    angles of the joints of the right arm
        :type   angles_l    np.array(dtype=float)
        :type   angles_r    np.array(dtype=float)

        '''
        self.angles_l = angles_l
        self.angles_r = angles_r
        self.position_l, _ = self.arm_l.get_joint_positions(self.angles_l)
        self.position_r, _ = self.arm_r.get_joint_positions(self.angles_r)

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def reset(self):
        '''
        Reset all angles to initial position
        '''
        self.set_angles(self.angles_l * 0.0, self.angles_r * 0.0)
        self.position_l, _ = self.arm_l.get_joint_positions(self.angles_l)
        self.position_r, _ = self.arm_r.get_joint_positions(self.angles_r)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


class BodySimulator(object):
    '''
    Control sensory inputs and motor actuators
    '''

    def __init__(self, pixels, lims, **kargs):
        '''
        :param  pixels                  width and length of the retina
        :param  lims                    x and y limits of the field of view
        :type   pixels                  list - [width, height]
        :type   lims                    list of pairs   - [[xmin, xmax], [ymin, ymax]]
        '''

        self.pixels = pixels
        self.lims = lims

        # init actuators
        self.actuator = KinematicActuator()
        self.theoric_actuator = KinematicActuator()
        self.target_actuator = KinematicActuator()

        self.actuator.reset()
        self.theoric_actuator.reset()
        self.target_actuator.reset()

        # init all data members
        n_joints = self.actuator.NUMBER_OF_JOINTS
        self.larm_angles = np.zeros(n_joints)
        self.rarm_angles = np.zeros(n_joints)
        self.larm_delta_angles = np.zeros(n_joints)
        self.rarm_delta_angles = np.zeros(n_joints)
        self.larm_angles_theoric = self.larm_angles
        self.rarm_angles_theoric = self.rarm_angles
        self.larm_angles_target = self.larm_angles
        self.rarm_angles_target = self.rarm_angles

        self.pos = np.zeros(self.pixels)
        self.pos_delta = np.zeros(self.pixels)
        self.pos_old = np.zeros(self.pixels)

        self.prop = np.zeros(self.pixels)
        self.prop_delta = np.zeros(self.pixels)
        self.prop_old = np.zeros(self.pixels)

        self.touch = np.zeros(self.pixels)
        self.touch_delta = np.zeros(self.pixels)
        self.touch_old = np.zeros(self.pixels)

        # initial body points
        self.init_body_tokens = (
            self.actuator.position_l[::-1],
            self.actuator.position_r)

        # init the perception manager
        self.perc = PerceptionManager(
            lims=lims,
            pixels=pixels,
            **kargs)

        #
        self.touches = np.zeros(len(self.perc.sensors))
        self.curr_body_tokens = self.init_body_tokens
        self.prev_body_tokens = self.init_body_tokens

        self.collisionManager = KM.CollisionManager()

    def get_positions(self,
                      langles,       rangles,
                      dlangles=None, drangles=None,
                      tlangles=None, trangles=None):

        self.actuator.set_angles(langles, rangles)
        if dlangles is not None and drangles is not None:
            self.theoric_actuator.set_angles(dlangles, drangles)
        if tlangles is not None and trangles is not None:
            self.target_actuator.set_angles(tlangles, trangles)


    def reset_body_chain(self):

        self.curr_body_tokens = self.prev_body_tokens[:]

    def single_step_forward(self):

        # compute actual angles
        self.actuator.set_angles(self.larm_angles, self.rarm_angles)
        self.theoric_actuator.set_angles(self.larm_angles_theoric,
                                         self.rarm_angles_theoric)
        self.target_actuator.set_angles(self.larm_angles_target,
                                        self.rarm_angles_target)

        # compute actual positions given the current angles
        self.get_positions(self.larm_angles, self.rarm_angles,
                           self.larm_angles_theoric, self.rarm_angles_theoric,
                           self.larm_angles_target, self.rarm_angles_target)

    def single_step_backward(self, count_substeps, delta, substeps):

        # try angles that are a little bit moved back
        # from those producing collision

        # go back of a fraction of angle
        larm_angles = (self.larm_angles
                       - count_substeps
                       * delta * (count_substeps**0.5)
                       * self.larm_delta_angles
                       / np.abs(self.larm_delta_angles)
                       / float(substeps * 2))
        rarm_angles = (self.rarm_angles
                       - count_substeps
                       * delta * (count_substeps**0.5)
                       * self.rarm_delta_angles
                       / np.abs(self.rarm_delta_angles)
                       / float(substeps * 2))

        # compute actual positions given the current angles
        self.get_positions(larm_angles, rarm_angles,
                           self.larm_angles_theoric, self.rarm_angles_theoric,
                           self.larm_angles_target, self.rarm_angles_target)

        self.get_body_chain(update_prev=False)

        return np.vstack(self.curr_body_tokens)

    def get_body_chain(self, update_prev=True):

        if update_prev == True:
            self.prev_body_tokens = self.curr_body_tokens[:]

        self.larm_angles, self.rarm_angles = (self.actuator.angles_l,
                                              self.actuator.angles_r)

        self.curr_body_tokens = (self.actuator.position_l[::-1],
                                 self.actuator.position_r)

    def step_kinematic(self,
                       larm_angles_unscaled, rarm_angles_unscaled,
                       larm_angles_theoric_unscaled, rarm_angles_theoric_unscaled,
                       larm_angles_target_unscaled, rarm_angles_target_unscaled,
                       active=True):
        '''
        :param  larm_angles_unscaled             actual angles of the joints of the left arm
        :param  rarm_angles_unscaled             actual angles of the joints of the right arm
        :param  larm_angles_theoric_unscaled     motor commands to the left arm
        :param  rarm_angles_theoric_unscaled     motor commands to the right arm
        :param  larm_angles_target_unscaled      desired angle end-point positions the left arm
        :param  rarm_angles_target_unscaled      desired angle end-point positions the right arm
        :param  active                           if the body_simulator is currently activa

        '''

        larm_angles, rarm_angles = self.actuator.rescale_angles(
            larm_angles_unscaled,
            rarm_angles_unscaled)

        larm_angles_theoric, rarm_angles_theoric = self.actuator.rescale_angles(
            larm_angles_theoric_unscaled,
            rarm_angles_theoric_unscaled)

        larm_angles_target, rarm_angles_target = self.actuator.rescale_angles(
            larm_angles_target_unscaled,
            rarm_angles_target_unscaled)

        # update previous data

        self.pos_old = self.pos
        self.prop_old = self.prop
        self.touch_old = self.touch

        # update current data
        self.larm_delta_angles = larm_angles - self.larm_angles
        self.rarm_delta_angles = rarm_angles - self.rarm_angles
        self.larm_angles = larm_angles
        self.rarm_angles = rarm_angles
        self.larm_angles_theoric = larm_angles_theoric
        self.rarm_angles_theoric = rarm_angles_theoric
        self.larm_angles_target = larm_angles_target
        self.rarm_angles_target = rarm_angles_target

        self.single_step_forward()
        self.get_body_chain()

        # compute collisions
        autocollision, _ = self.collisionManager.manage_collisions(
            prev_chain=np.vstack(self.prev_body_tokens),
            curr_chain=np.vstack(self.curr_body_tokens),
            move_back_fun=self.single_step_backward,
            reset=None)

        #
        # # control collision resolution
        # larm_angles = self.larm_angles
        # rarm_angles = self.rarm_angles
        #
        # count_collisions = 1
        # c_scale = 20
        # delta = np.pi * (1.0 / 15.0)
        #
        # while autocollision:
        #     if count_collisions < c_scale:
        #
        #         # try angles that are a little bit moved back
        #         # from those producing collision
        #
        #         # go back of a fraction of angle
        #         larm_angles = (self.larm_angles
        #                        - count_collisions
        #                        * delta * (count_collisions**0.5)
        #                        * self.larm_delta_angles
        #                        / np.abs(self.larm_delta_angles)
        #                        / float(c_scale * 2))
        #         rarm_angles = (self.rarm_angles
        #                        - count_collisions
        #                        * delta * (count_collisions**0.5)
        #                        * self.rarm_delta_angles
        #                        / np.abs(self.rarm_delta_angles)
        #                        / float(c_scale * 2))
        #
        #         # compute actual positions given the current angles
        #         self.get_positions(larm_angles, rarm_angles,
        #                            self.larm_angles_theoric, self.rarm_angles_theoric,
        #                            self.larm_angles_target, self.rarm_angles_target)
        #
        #         # compute collisions
        #         autocollision = self.get_collision()
        #
        #         count_collisions += 1
        #
        #     else:
        #
        #         # if still colliding after 10 trials
        #         # give up and reset to the angles before colliding
        #
        #         larm_angles = self.larm_angles - self.larm_delta_angles / 2.0
        #         rarm_angles = self.rarm_angles - self.rarm_delta_angles / 2.0
        #
        #         # compute actual positions given the current angles
        #         self.get_positions(larm_angles, rarm_angles,
        #                            self.larm_angles_theoric, self.rarm_angles_theoric,
        #                            self.larm_angles_target, self.rarm_angles_target)
        #
        #         autocollision = False
        #         res_autocollision = False

        self.larm_angles = larm_angles
        self.rarm_angles = rarm_angles

        #######################################################################

        # VISUAL POSITION
        self.pos = self.perc.get_image(body_tokens=self.curr_body_tokens)

        # PROPRIOCEPTION
        angles_tokens = (self.larm_angles, self.rarm_angles)
        self.prop = self.perc.get_proprioception(
            angles_tokens=angles_tokens)
        # TOUCH
        self.touch, self.touches = self.perc.get_touch(
            body_tokens=self.curr_body_tokens)

        delta_angles_tokens = (self.larm_delta_angles,
                               self.rarm_delta_angles)

        # deltas
        self.pos_delta = self.perc.get_image(
            body_tokens=self.curr_body_tokens)
        self.prop_delta = self.perc.get_proprioception(
            angles_tokens=delta_angles_tokens)
        self.touch_delta = self.touch - self.touch_old

        return active and autocollision

    def reset(self):
        self.pos_old *= 0
        self.pos *= 0
        self.prop_old *= 0
        self.prop *= 0
        self.touch_old *= 0
        self.touch *= 0
