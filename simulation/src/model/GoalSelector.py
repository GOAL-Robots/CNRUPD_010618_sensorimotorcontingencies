#!/usr/bin/env python

import numpy as np
from model.esn import ESN
import model.kinematics as KM
from gauss_utils import OptimizedGaussianMaker as GaussianMaker
from gauss_utils import pseudo_diag


def softmax(x, t=0.1):
    '''
    :param x: array of values
    :param t: temperature
    :return: softmax of values
    '''
    e = np.exp(x / t)
    return e / np.sum(e)


def my_argwhere(x):

    res = np.nonzero(x)[0]

    return res


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
        
        res = -(np.sin(2.0 * np.pi * self.freqs * self.t / 
                       self.scale + np.pi)) * 0.5

        self.t += 1

        return res

    def reset(self):

        self.t = 0


def oscllator(x, scale, freqs):
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


class GoalSelector(object):

    def __init__(self, dt, tau, alpha, epsilon, eta, n_input,
                 n_goal_units, n_echo_units, n_rout_units,
                 match_decay, noise, scale, sm_temp, g2e_spars,
                 goal_window, goal_learn_start, reset_window, echo_ampl=1000,
                 multiple_echo=True):
        '''
        :param dt: integration time of the ESN
        :param tau: decay of the ESN
        :param alpha: infinitesimal rotation coefficent of the ESN
        :param epsilon:  ESN distance from the unit spectral radius
        :param eta: learning rate
        :param n_input: number of inputs
        :param n_goal_units: number of units in the goal layer
        :param n_echo_units: number of units in the ESN
        :param n_rout_units: number of actuators
        :param match_decay: decay of the matching trace
        :param noise: standard deviation of white noise in the actuators
        :param scale: scale of the noise oscillator periods
        :param sm_temp: temperature of the softmax
        :param g2e_spars: sparseness of weights form goals to esn
        :param goal_window: max duration of a goal selection
        :param goal_learn_start: start of learning during trial
        :param reset_window: duration of reset
        :param echo_ampl: amplitude of the input to the echo-state
        :param multiple_echo: if True each goal calls its own esn
        '''

        self.DT = dt
        self.TAU = tau
        self.ALPHA = alpha
        self.EPSILON = epsilon
        self.ETA = eta
        self.MATCH_DECAY = match_decay
        self.NOISE = noise
        self.scale = scale
        self.SM_TEMP = sm_temp
        self.GOAL_WINDOW = goal_window
        self.GOAL_LEARN_START = goal_learn_start
        self.RESET_WINDOW = reset_window

        self.N_INPUT = n_input
        self.N_GOAL_UNITS = n_goal_units
        self.N_ECHO_UNITS = n_echo_units
        self.N_ROUT_UNITS = n_rout_units
        self.GOAL2ECHO_SPARSENESS = g2e_spars
        self.ECHO_AMPL = echo_ampl
        self.MULTIPLE_ECHO = multiple_echo

        self.goal_selection_vec = np.zeros(self.N_GOAL_UNITS)
        self.goal_window_counter = 0
        self.reset_window_counter = 0

        self.echonet = [
            ESN(
                N=self.N_ECHO_UNITS,
                stime=self.GOAL_WINDOW,
                dt=self.DT,
                tau=self.TAU,
                alpha=self.ALPHA,
                beta=1 - self.ALPHA,
                epsilon=self.EPSILON
            ) for x in xrange(self.N_GOAL_UNITS + 1)]

        #----------------------------------------------------
        # input -> ESN

        # create weight matrix
        esn_input_units = self.N_ECHO_UNITS / 4
        cols = self.N_INPUT + self.N_GOAL_UNITS
        rows = self.N_ECHO_UNITS
        self.INP2ECHO_W = np.zeros([rows, cols])

        # set all weights to random values
        self.INP2ECHO_W[:esn_input_units, :] = \
            0.0*np.random.randn(esn_input_units,
                            self.N_INPUT + self.N_GOAL_UNITS)

        # weights are sparse, set all other weights to zero
        self.INP2ECHO_W[:esn_input_units, :] *= \
            (np.random.rand(esn_input_units,
                            self.N_INPUT + self.N_GOAL_UNITS) < self.GOAL2ECHO_SPARSENESS)

        #----------------------------------------------------
        # goal_layer -> ESN

        # create weight matrix
        self.GOAL2ECHO_W = np.zeros([self.N_ECHO_UNITS,
                                     self.N_GOAL_UNITS])

        dim = int(np.sqrt(self.N_GOAL_UNITS))
        lims = [[0, dim - 1, dim],
                [0, dim - 1, dim]]
        gm = GaussianMaker(lims)
        goal_filter_weights_sigma = 1.5 / float(dim)
        self.GOAL_FILTER_WEIGHTS = np.vstack([
            [gm((x, y), goal_filter_weights_sigma)[0]
             for x in xrange(dim)]
            for y in xrange(dim)])

        half_weights = pseudo_diag(self.N_ECHO_UNITS / 2,  self.N_GOAL_UNITS)

        # set first half of weight rows to negative values
        self.GOAL2ECHO_W[:(self.N_ECHO_UNITS / 2), :] = -half_weights/float(self.N_GOAL_UNITS)

        # set second half of weight rows to positive random values
        self.GOAL2ECHO_W[(self.N_ECHO_UNITS / 2):, :] = half_weights/float(self.N_GOAL_UNITS)


        # # set first half of weight rows to negative random values
        # self.GOAL2ECHO_W[:(self.N_ECHO_UNITS / 2), :] = \
        #     -np.random.rand(self.N_ECHO_UNITS / 2,
        #                     self.N_GOAL_UNITS)
        #
        # # weights are sparse, set other weights of the first half to zero
        # self.GOAL2ECHO_W[:(self.N_ECHO_UNITS / 2), :] *= \
        #     (np.random.rand(self.N_ECHO_UNITS / 2,
        #                     self.N_GOAL_UNITS) < self.GOAL2ECHO_SPARSENESS)
        #
        # # set second half of weight rows to positive random values
        # self.GOAL2ECHO_W[(self.N_ECHO_UNITS / 2):, :] = \
        #     np.random.rand(self.N_ECHO_UNITS / 2,
        #                    self.N_GOAL_UNITS)
        #
        # # weights are sparse, set other weights of the second half to zero
        # self.GOAL2ECHO_W[(self.N_ECHO_UNITS / 2):, :] *= \
        #     (np.random.rand(self.N_ECHO_UNITS / 2,
        #                     self.N_GOAL_UNITS) < self.GOAL2ECHO_SPARSENESS)

        #----------------------------------------------------
        # ESN readouts

        self.echo2out_w = [
            0.1 * np.random.randn(self.N_ROUT_UNITS,
                                  self.N_ECHO_UNITS) for x in
            xrange(self.N_GOAL_UNITS * self.MULTIPLE_ECHO + 1)]

        self.read_out = np.zeros(self.N_ROUT_UNITS)
        self.out = np.zeros(self.N_ROUT_UNITS)
        self.tout = np.zeros(self.N_ROUT_UNITS)
        self.gout = np.zeros(self.N_ROUT_UNITS)
        self.target_position = dict()
        self.target_counter = dict()
        self.match_mean = np.zeros(self.N_GOAL_UNITS)
        self.curr_noise = 0.0

        self.is_goal_selected = False
        self.reset_oscillator()
        self.t = 0

        self.curr_echonet = self.echonet[-1]
        self.curr_echo2out_w = self.echo2out_w[-1]

    def reset_oscillator(self):
        self.random_oscil = np.random.rand(self.N_ROUT_UNITS)
        self.oscillator = Oscillator(self.scale, self.random_oscil)

    def goal_index(self):

        if np.sum(self.goal_selection_vec) > 0:
            idx = np.nonzero(self.goal_selection_vec > 0)[0][0]
            return idx

        return None

    def goal_selection(self, goal_mask=None,
                       competence_improvement_vec=None,
                       incompetence_vec=None):
        '''
        :param im_value: current intrinsic motivational value
        :param goal_mask: which goals can be selected
        :param incompetence_vec: which goals are more novel to the controller
        '''

        # in case we do not have a mask create one
        # ans select all goals as possible
        if goal_mask is None:
            goal_mask = np.ones(self.N_GOAL_UNITS)

        # if no goal has been selected
        if self.is_goal_selected == False:

            # get indices of the currently avaliable goals
            curr_goal_idcs = my_argwhere(goal_mask > 0)

            #------------------------------------------------------------
            # compose motivations

            motivation = np.zeros(curr_goal_idcs.shape)

            # get values of the averages competence improovements
            # of the currently avaliable goals
            if competence_improvement_vec is not None:
                motivation = competence_improvement_vec[curr_goal_idcs]

            # add incompetence
            if incompetence_vec is not None:
                motivation += incompetence_vec[curr_goal_idcs]

            #------------------------------------------------------------

            # compute softmax between the currently avaliable goals
            self.sm = softmax(motivation, self.SM_TEMP)

            # cumulate probabilities between them
            cum_prob = np.hstack((0, np.cumsum(self.sm)))

            # flip the coin
            coin = np.random.rand()

            # the winner between avaliable goals based on the flipped coin
            cur_goal_win = np.logical_and(cum_prob[:-1] < coin,
                                          cum_prob[1:] >= coin)

            # index of the winner in the vector of all goals
            goal_win_idx = curr_goal_idcs[my_argwhere(cur_goal_win == True)]

            # reset goal_selection_vec to all False values
            self.goal_selection_vec *= False

            # set the winner to True
            self.goal_selection_vec[goal_win_idx] = True

            self.t = 0
            self.reset_oscillator()

            self.is_goal_selected = True

            goalwin_idx = self.goal_index()

            MULTIPLE_ECHO = self.MULTIPLE_ECHO

            if MULTIPLE_ECHO == True and goalwin_idx is not None:
                self.curr_echonet = self.echonet[goalwin_idx]
                self.curr_echo2out_w = self.echo2out_w[goalwin_idx]
            else:
                self.curr_echonet = self.echonet[-1]
                self.curr_echo2out_w = self.echo2out_w[-1]

            if goalwin_idx is not None:
                if self.target_position.has_key(goalwin_idx):
                    target = self.target_position[goalwin_idx]
                    self.gout = target
            else:
                self.gout = np.zeros(self.N_ROUT_UNITS)

    def update_target(self, curr_pos):

        goalwin_idx = self.goal_index()

        if goalwin_idx is not None:

            self.target_counter[goalwin_idx] = (
                self.target_counter.setdefault(goalwin_idx, 0) + 1)
            self.target_counter[goalwin_idx] = 1
            pos = curr_pos
            self.target_position.setdefault(goalwin_idx,  pos)
            pos_mean = self.target_position[goalwin_idx]
            n = self.target_counter[goalwin_idx]
            pos_mean += (1.0 -
                         self.match_mean[self.goal_selection_vec > 0]) * (-pos_mean + pos)
            self.target_position[goalwin_idx] = pos_mean

    def reset(self, match):

        self.match_mean += self.MATCH_DECAY * (
            -self.match_mean + match) * self.goal_selection_vec

        self.curr_echonet.reset()
        self.curr_echonet.reset_data()

        self.goal_selection_vec *= 0
        self.goal_window_counter = 0
        self.reset_window_counter = 0

        self.out *= 0

    def getCurrMatch(self):

        res = self.match_mean[self.goal_selection_vec > 0]
        if len(res) == 1:
            return np.asscalar(res)
        return 0

    def step(self, inp):
        '''
        :param inp: external input
        '''

        goal2echo_inp = np.dot(
            self.GOAL2ECHO_W,
            np.dot(self.GOAL_FILTER_WEIGHTS,
                self.goal_selection_vec) * self.is_goal_selected)

        inp2echo_inp = np.dot(
            self.INP2ECHO_W,
            np.hstack((
                inp,
                self.goal_selection_vec * self.is_goal_selected
            )))

        goalwin_idx = self.goal_index()

        echo_inp = (inp2echo_inp +  goal2echo_inp)
        self.curr_echonet.step(self.ECHO_AMPL * echo_inp)
        self.curr_echonet.store(self.goal_window_counter)

        self.inp = self.curr_echonet.out
        self.read_out = np.tanh(np.dot(self.curr_echo2out_w,
                                       self.curr_echonet.out))

        curr_match = self.getCurrMatch()

        if np.all(self.goal_selection_vec == 0):
            curr_match = 0.0

        # OSCILLATOR NOISE
        added_signal = self.NOISE * self.oscillator()

        self.out = curr_match * self.read_out + \
            (1.0 - curr_match) * added_signal

        self.tout = self.read_out

        self.t += 1

    def learn(self):

        goalwin_idx = self.goal_index()

        #------------------------------------------------
        if (goalwin_idx is not None and
                self.target_position.has_key(goalwin_idx)):
            if self.target_position.has_key(goalwin_idx):
                target = self.target_position[goalwin_idx]
                x = self.inp
                y = self.tout
                eta = self.ETA
                w = self.curr_echo2out_w
                w += eta * np.outer((1 - y**2) * (target - y), x)
        #------------------------------------------------
