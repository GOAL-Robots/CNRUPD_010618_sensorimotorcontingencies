#!/usr/bin/env python
import numpy as np
from model import *
from model.kohonen import Kohonen

def step_fun(x, th):
    return 1.0 * (x > th)

def normalize(x) :
    nrm = np.linalg.norm(x)
    if nrm > 0:
        return x/np.linalg.norm(x)
    return np.zeros(x.shape)

class GoalMaker(object):
    def __init__(self, n_input_layers, n_singlemod_layers, n_hidden_layers,
            n_out, n_goalrep,singlemod_lrs, hidden_lrs, output_lr, goalrep_lr,
            goal_th=0.9, single_kohonen=False, stime=1000):
        '''
        :param n_input_layers: (list) number of units per input
        :param n_singlemod_layers: (list) number of units per singlemod layer
        :param n_hidden_layers: (list) number of units per hidden layer
        :param n_out: (int) number of output units
        :param n_goalrep: (int) number of goal representation units
        :param singlemod_lrs: (list) learning rates of single modality networks
        :param hidden_lrs: (list) learning rates of hidden networks
        :param output_lr: (float) learning rate of output network
        :param goalrep_lr: (float) learning rate goalrep som
        :param goal_th: (float) goal formation threshold
        :param single_kohonen: (bool) if the internal net is to be reduced to a single kohonen
        :param stime: (int) simulation timesteps
        '''

        self.TOT_INPUT_LAYERS = len(n_input_layers)
        self.N_INPUT_LAYERS = n_input_layers
        self.TOT_SINGLEMOD_LAYERS = len(n_singlemod_layers)
        self.N_SINGLEMOD_LAYERS = n_singlemod_layers
        self.TOT_HIDDEN_LAYERS = len(n_hidden_layers)
        self.N_HIDDEN_LAYERS = n_hidden_layers
        self.N_OUTPUT_LAYER = n_out
        self.N_GOALREP_LAYER = n_goalrep
        self.GOAL_TH = goal_th
        self.SINGLE_KOHONEN = single_kohonen
        self.STIME = stime

        self.input_layers = [np.zeros(x) for x in self.N_INPUT_LAYERS]
        self.prev_raw_inputs = [np.zeros(x) for x in self.N_INPUT_LAYERS]
        self.singlemod_layers = [np.zeros(x) for x in self.N_SINGLEMOD_LAYERS]
        self.hidden_layers = [np.zeros(x) for x in self.N_HIDDEN_LAYERS]
        self.output_layer = np.zeros(self.N_OUTPUT_LAYER)
        self.goalrep_layer = np.zeros(self.N_GOALREP_LAYER)

        # init SOM networks

        # SINGLE MODALITY LAYERS
        self.singlemod_soms = []
        for n_singlemod in xrange(self.TOT_SINGLEMOD_LAYERS):
            n_input = self.N_INPUT_LAYERS[n_singlemod]
            n_output = self.N_SINGLEMOD_LAYERS[n_singlemod]
            net = Kohonen(
                n_output = n_output,
                n_input = n_input,
                n_dim_out = gm_singlemod_n_dim_out,
                bins = [np.sqrt(n_output), np.sqrt(n_output)],
                eta = singlemod_lrs[n_singlemod],
                eta_bl = singlemod_lrs[n_singlemod]/gm_singlemod_eta_bl_scale,
                eta_decay = self.STIME/gm_singlemod_eta_decay_scale,
                neighborhood = gm_singlemod_neigh_scale*n_output,
                neighborhood_decay = self.STIME/gm_singlemod_neigh_decay_scale,
                neighborhood_bl = gm_singlemod_neigh_bl,
                stime = self.STIME,
                weight_bl = gm_singlemod_weight_bl,
                normalize = normalize
            )
            self.singlemod_soms.append(net)

        # HIDDEN
        self.hidden_soms = []
        for n_hidden in xrange(self.TOT_HIDDEN_LAYERS):
            n_input = sum(self.N_SINGLEMOD_LAYERS[n_hidden:(n_hidden + 2)])
            n_output = self.N_HIDDEN_LAYERS[n_hidden]
            net = Kohonen(
                n_output = n_output,
                n_input = n_input,
                n_dim_out = gm_hidden_n_dim_out,
                bins = [np.sqrt(n_output), np.sqrt(n_output)],
                eta = hidden_lrs[n_hidden],
                eta_bl = hidden_lrs[n_hidden]/gm_hidden_eta_bl_scale,
                eta_decay = self.STIME/gm_hidden_eta_decay_scale,
                neighborhood = gm_hidden_neigh_scale*n_output,
                neighborhood_decay = self.STIME/gm_hidden_neigh_decay_scale,
                neighborhood_bl = gm_hidden_neigh_bl,
                stime = self.STIME,
                weight_bl = gm_hidden_weight_bl,
                normalize = normalize
            )
            self.hidden_soms.append(net)

        # OUTPUT
        n_input = sum(self.N_HIDDEN_LAYERS)
        n_output = self.N_OUTPUT_LAYER
        self.out_som = Kohonen(
            n_output = n_output,
            n_input = n_input,
            n_dim_out = gm_out_n_dim_out,
            bins = [np.sqrt(n_output), np.sqrt(n_output)],
            eta = output_lr,
            eta_bl = output_lr/gm_out_eta_bl_scale,
            eta_decay = self.STIME/gm_out_eta_decay_scale,
            neighborhood = gm_out_neigh_scale*n_output,
            neighborhood_decay = self.STIME/gm_out_neigh_decay_scale,
            neighborhood_bl = gm_out_neigh_bl,
            stime = self.STIME,
            weight_bl = gm_out_weight_bl,
            normalize = normalize
        )

        # GOALS
        if self.SINGLE_KOHONEN == True :
            n_input =  self.N_INPUT_LAYERS[-1] # last input layer size (touch)
            n_output = self.N_GOALREP_LAYER
            self.goalrep_som = Kohonen(
                n_output = n_output,
                n_input = n_input,
                n_dim_out = gm_single_kohonen_n_dim_out,
                bins =[np.sqrt(n_output), np.sqrt(n_output)],
                eta = goalrep_lr,
                eta_bl = gm_single_kohonen_eta_bl,
                eta_decay = gm_single_kohonen_eta_decay,
                neighborhood = gm_single_kohonen_neigh_scale*np.sqrt(n_output),
                neighborhood_decay = gm_single_kohonen_neigh_decay_scale,
                neighborhood_bl = gm_single_kohonen_neigh_bl,
                stime =self.STIME,
                weight_bl = gm_single_kohonen_weight_bl,
                normalize = normalize
            )

        else:

            n_input = sum(self.N_SINGLEMOD_LAYERS) \
                    + sum(self.N_HIDDEN_LAYERS) + self.N_OUTPUT_LAYER
            n_output = self.N_GOALREP_LAYER
            self.goalrep_som = Kohonen(
                n_output = n_output,
                n_input = n_input,
                n_dim_out = gm_goalrep_n_dim_out,
                bins = n_output,
                eta = goalrep_lr,
                eta_bl = goalrep_lr/gm_goalrep_eta_bl_scale,
                eta_decay = self.STIME/gm_goalrep_eta_decay_scale,
                neighborhood = gm_goalrep_neigh_scale*n_output,
                neighborhood_decay = self.STIME/gm_goalrep_neigh_decay_scale,
                neighborhood_bl = gm_goalrep_neigh_bl,
                stime = self.STIME,
                weight_bl = gm_goalrep_weight_bl,
                normalize = normalize
            )

    def step(self, raw_inputs, neigh_scale=None):
        '''
        :param raw_inputs: (list) raw inputs in all modalities
        :param neigh_scale: manual value of the neighborhood radius
        '''

        assert len(raw_inputs) == self.TOT_INPUT_LAYERS, \
            """
            Input modalities must be of the same
            number of the input layers
            """

        # Derivatives
        for layer in xrange(self.TOT_INPUT_LAYERS):
            self.input_layers[layer] = \
                np.maximum(0,raw_inputs[layer] - self.prev_raw_inputs[layer])

        # PREPARE INPUTS

        if self.SINGLE_KOHONEN == False :

            # singlemod

            n_singlemod_inp = []
            for n_singlemod in xrange(self.TOT_INPUT_LAYERS):
                n_singlemod_inp.append(self.input_layers[n_singlemod])

            # hidden

            n_hidden_inp = []
            for n_hidden in xrange(self.TOT_SINGLEMOD_LAYERS):
                n_hidden_inp.append(np.hstack(
                    self.singlemod_layers[n_hidden:(n_hidden + 2)]))

            # output

            out_inp = np.hstack(self.hidden_layers)

            # goal representation
            goalrep_inp = np.hstack([
                np.hstack(self.singlemod_layers),
                np.hstack(self.hidden_layers),
                self.output_layer])

            # SPREADING

            # input to single modality
            for n_singlemod in xrange(self.TOT_SINGLEMOD_LAYERS):
                self.singlemod_soms[n_singlemod].step(n_singlemod_inp[n_singlemod])

            # single modality to hidden
            for n_hidden in xrange(self.TOT_HIDDEN_LAYERS):
                self.hidden_soms[n_hidden].step(n_hidden_inp[n_hidden])

            # hidden to output
            self.out_som.step(out_inp)

            # UPDATE ACTIVATIONS

            # single modality (The activation is multiplied
            # times the highest unit of the som)
            for n_singlemod in xrange(self.TOT_SINGLEMOD_LAYERS):
                self.singlemod_layers[n_singlemod] = step_fun(
                    (self.singlemod_soms[n_singlemod].out == 1) *
                    self.singlemod_soms[n_singlemod].out_raw,
                    self.GOAL_TH)

            # hidden (The activation is multiplied
            # times the highest unit of the som)
            for n_hidden in xrange(self.TOT_HIDDEN_LAYERS):
                self.hidden_layers[n_hidden] = step_fun(
                    (self.hidden_soms[n_hidden].out == 1) *
                    self.hidden_soms[n_hidden].out_raw,
                    self.GOAL_TH)

            # output (The activation is multiplied
            # times the highest unit of the som)
            self.output_layer = \
            step_fun((self.out_som.out == 1) *\
                    self.out_som.out_raw, self.GOAL_TH)

        else :
            # goal representation
            goalrep_inp = np.hstack(self.input_layers[-1]) # last layer is touch

        # all to goal rep
        self.goalrep_som.step(goalrep_inp,neigh_scale = neigh_scale)

        # goal rep (The activation is multiplied
        # times the highest unit of the som)
        self.goalrep_layer = \
                step_fun((self.goalrep_som.out == 1) *\
                self.goalrep_som.out_raw, self.GOAL_TH)

        # previous input storing
        for layer in xrange(self.TOT_INPUT_LAYERS):
            self.prev_raw_inputs[layer] = raw_inputs[layer]

    def learn(self, eta_scale=None, pred=None):
        for som in self.singlemod_soms:
            som.learn(eta_scale=eta_scale)
        for som in self.hidden_soms:
            som.learn(eta_scale=eta_scale)
        self.out_som.learn(eta_scale=eta_scale)
        self.goalrep_som.learn(eta_scale=eta_scale, pred=pred)

#----------------------------------------------------------------

class GoalTesterSim(object):

    def __init__(self):

        self.goalmaker = GoalMaker(
            n_input_layers=[400, 400, 400],
            n_singlemod_layers= [64, 64, 64],
            n_hidden_layers=[16, 16],
            n_out=16,
            n_goalrep= 64,
            singlemod_lrs = [0.05, 0.05, 0.05],
            hidden_lrs=[0.01, 0.01],
            output_lr=0.005,
            goalrep_lr=0.2,
            goal_th=0.1
        )

        self.stime = 100000
        self.body_simulator = Simulator.BodySimulator()

    def step(self,t):
        '''
        :param t: timestep
        :return: tuple with current values of all activations and weights
        '''

        period = 10.0
        if np.mean(self.body_simulator.actuator.angles_l) < np.pi*0.2 :
            ldelta = ( np.ones(self.body_simulator.actuator.NUMBER_OF_JOINTS) *
                      0.3+np.sin(t/period)*0.2 )
        else :
            ldelta = ( -np.ones(self.body_simulator.actuator.NUMBER_OF_JOINTS) *
                      0.3+np.sin(t/period)*0.2 )

        if np.mean(self.body_simulator.actuator.angles_r) < np.pi*0.2 :
            rdelta = ( np.ones(self.body_simulator.actuator.NUMBER_OF_JOINTS) *
                      0.2+np.sin(t/period)*0.2 )
        else :
            rdelta = (-np.ones(self.body_simulator.actuator.NUMBER_OF_JOINTS) *
                      0.2+np.sin(t/period)*0.2 )

        self.body_simulator.step(larm_delta_angles=ldelta,
                                 rarm_delta_angles=rdelta)

        self.goalmaker.step([self.body_simulator.pos_delta.ravel(),
                             self.body_simulator.prop_delta.ravel()*50.0,
                             self.body_simulator.touch_delta.ravel()*50.0])

        self.goalmaker.learn()


        i1, i2, i3 = self.goalmaker.input_layers
        sm1, sm2, sm3 = self.goalmaker.singlemod_layers
        h1, h2 = self.goalmaker.hidden_layers
        wsm1, wsm2, wsm3 = (som.inp2out_w for som in
                            self.goalmaker.singlemod_soms)
        wh1, wh2 = (som.inp2out_w for som in self.goalmaker.hidden_soms)
        wo1 = self.goalmaker.out_som.inp2out_w
        o1 = self.goalmaker.output_layer
        wgr1 = self.goalmaker.goalrep_som.inp2out_w
        gr1 = self.goalmaker.goalrep_layer

        return i1, i2, i3, sm1, sm2, sm3, h1, h2, o1, wsm1, \
                wsm2, wsm3, wh1, wh2, wo1, wgr1, gr1
