import numpy as np
from model import *
from model.GoalPredictor import match

def linear(x, x_low, x_up, y_low=0, y_up=1):
    
    return  np.maximum(y_low, np.minimum(y_up,
        ((y_up - y_low) / (x_up - x_low)) * 
        (x - x_low)))

class Simulation(object) :

    def __init__(self, rng=None) :
        
        self.rng = rng
        if rng is None:
            self.seed = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
            self.rng = np.random.RandomState(self.seed) 

        self.GOAL_NUMBER = GOAL_NUMBER

        self.body_simulator = BodySimulator(
                pixels=body_simulator_pixels,
                lims=body_simulator_lims,
                touch_th=body_simulator_touch_th,
                num_touch_sensors=body_simulator_num_touch_sensors,
                touch_sigma=body_simulator_touch_sigma,
                )

        self.gs = GoalSelector(
                dt=gs_dt,
                tau=gs_tau,
                alpha=gs_alpha,
                epsilon=gs_epsilon,
                eta=gs_eta,
                n_input=gs_n_input,
                n_goal_units=gs_n_goal_units,
                n_echo_units=gs_n_echo_units,
                n_rout_units=gs_n_rout_units,
                match_decay=gs_match_decay,
                noise=gs_noise,
                scale=gs_noise_scale,
                sm_temp=gs_sm_temp,
                g2e_spars=gs_g2e_spars,
                echo_ampl=gs_echo_ampl,
                goal_window=gs_goal_window,
                goal_learn_start=gs_goal_learn_start,
                reset_window=gs_reset_window,
                multiple_echo=gs_multiple_echo,
                rng=self.rng
                )

        self.gp = GoalPredictor(
                n_goal_units=self.GOAL_NUMBER,
                eta=gp_eta
                )

        self.gm = GoalMaker(
                n_input_layers=gm_n_input_layers,
                n_singlemod_layers=gm_n_singlemod_layers,
                n_hidden_layers=gm_n_hidden_layers,
                n_out=gm_n_out,
                n_goalrep=gm_n_goalrep,
                singlemod_lrs=gm_singlemod_lrs,
                hidden_lrs=gm_hidden_lrs,
                output_lr=gm_output_lr,
                goalrep_lr=gm_goalrep_lr,
                goal_th=gm_goal_th,
                stime=gm_stime,
                single_kohonen=gm_single_kohonen
            )

        self.IM_AMP = simulation_im_amp

        self.IM_DECAY = simulation_im_decay

        self.COMPETENCE_IMPROVEMENT_PROP = (
            simulation_competence_improvement_prop)

        self.INCOMPETENCE_PROP = simulation_incompetence_prop

        self.stime = robot_stime

        self.trial_window = self.gs.RESET_WINDOW + self.gs.GOAL_WINDOW

        self.goal_mask = np.zeros(self.gs.N_GOAL_UNITS).astype("bool")

        self.competence_improvement_vec = np.zeros(self.gs.N_GOAL_UNITS)

        self.match_value = False

        self.intrinsic_motivation_value = 0.0

        self.static_inp = np.zeros(self.gs.N_INPUT)

        self.collision = False

        self.init_streams()
        
        self.timestep = 0
        self.kohonen_weights = None
        self.echo_weights = None
        
        self.current_target = None

    def init_streams(self):

        self.log_sensors = None
        self.log_cont_sensors = None
        self.log_position = None
        self.log_predictions = None
        self.log_targets = None
        self.log_weights = None

    def get_selection_arrays(self) :

        sel = self.gs.is_goal_selected

        gmask = self.goal_mask.astype("float")
        gv = self.competence_improvement_vec
        gw = self.gs.goal_selection_vec
        gr = self.gm.goalrep_layer

        goal_keys = self.gs.target_position.keys()

        acquired_targets = []
        if len(goal_keys) != 0:
            acquired_targets = self.gs.target_position.keys()

        matched_targets = np.array([ 1.0 * (target in acquired_targets)
            for target in np.arange(self.gs.N_GOAL_UNITS) ])

        targets = self.gp.w

        esn_data = self.gs.curr_echonet.data[self.gs.curr_echonet.out_lab]

        return (gmask, gv, gw, gr, matched_targets,
                targets, self.gs.target_position, esn_data)


    def get_sensory_arrays(self) :

        i1, i2, i3 = self.gm.input_layers
        sm1, sm2, sm3 = self.gm.singlemod_layers
        h1, h2 = self.gm.hidden_layers
        wsm1, wsm2, wsm3 = (som.inp2out_w for som in self.gm.singlemod_soms)
        wh1, wh2 = (som.inp2out_w for som in self.gm.hidden_soms)
        wo1 = self.gm.out_som.inp2out_w
        o1 = self.gm.output_layer
        wgr1 = self.gm.goalrep_som.inp2out_w
        gr1 = self.gm.goalrep_layer

        return (i1, i2, i3, sm1, sm2, sm3, h1,
                h2, o1, wsm1, wsm2, wsm3, wh1,
                wh2, wo1, wgr1, gr1)

    def get_arm_positions(self) :

        sel = self.gs.is_goal_selected

        real_l_pos = self.body_simulator.actuator.position_l
        real_r_pos = self.body_simulator.actuator.position_r

        real_l_pos *= sel
        real_r_pos *= sel

        goalwin_idx = self.gs.goal_index()
        target_l_pos = self.body_simulator.target_actuator.position_l
        target_r_pos = self.body_simulator.target_actuator.position_r
        if  not self.gs.target_position.has_key(goalwin_idx) :
            target_l_pos *= 0
            target_r_pos *= 0

        target_l_pos *= sel
        target_r_pos *= sel

        theor_l_pos = self.body_simulator.theoric_actuator.position_l
        theor_r_pos = self.body_simulator.theoric_actuator.position_r

        sensors = self.body_simulator.perc.sensors * sel

        return (real_l_pos, real_r_pos, target_l_pos,
                target_r_pos, theor_l_pos, theor_r_pos, sensors)

    def get_weights_metrics(self):

        res = dict()
        w = None

        # add norm of the kohonen weights
        if self.gm.SINGLE_KOHONEN :
            w = self.gm.goalrep_som.inp2out_w.ravel()
        else :
            w = np.hstack((
                np.hstack([som.inp2out_w.ravel
                           for som in self.gm.singlemod_soms ]),
                np.hstack([som.inp2out_w.ravel
                           for som in self.gm.hidden_soms ]),
                self.gm.out_som.inp2out_w.ravel(),
                self.gm.goalrep_som.inp2out_w.ravel()
                ))
        if self.kohonen_weights is None :
            res["kohonen_weights"] = np.linalg.norm(w)
        else:
            res["kohonen_weights"] = np.linalg.norm(w - self.kohonen_weights)

        self.kohonen_weights = w.copy()

        # add norm of the ESN weights
        w = self.gs.curr_echo2out_w
        if self.echo_weights is None :
            res["echo_weights"] = np.linalg.norm(w)
        else:
            res["echo_weights"] = np.linalg.norm(w - self.echo_weights)

        self.echo_weights = w.copy()

        return res

    def save_cont_logs(self) :

        if self.log_cont_sensors is not None :

            # save sensor info on file

            # create log line
            log_string = ""
            # add timing
            log_string += "{:8d} ".format(self.timestep)
            # add touch info
            for touch in  self.body_simulator.touches :
                log_string += "{:6.4f} ".format(touch)
            # add goal index
            log_string += "{:6d}".format(np.argmax(self.gs.goal_selection_vec))
            # save to file
            self.log_cont_sensors.write(log_string + "\n")
            self.log_cont_sensors.flush()

    def save_match_logs(self) :

        if self.log_sensors is not None :

            # save sensor info on file

            # create log line
            log_string = ""
            # add timing
            log_string += "{:8d} ".format(self.timestep)
            # add touch info
            for touch in  self.body_simulator.touches :
                log_string += "{:6.4f} ".format(touch)
            # add goal index
            log_string += "{:6d}".format(np.argmax(self.gs.goal_selection_vec))
            # save to file
            self.log_sensors.write(log_string + "\n")
            self.log_sensors.flush()

        if self.log_position is not None :

            # save position info on file

            # create log line
            log_string = ""
            # add timing
            log_string += "{:8d} ".format(self.timestep)
            # add position info
            curr_position = np.vstack(
                self.body_simulator.curr_body_tokens).ravel()
            for pos in  curr_position:
                log_string += "{:6.4f} ".format(pos)
            # add goal index
            log_string += "{:6d}".format(np.argmax(
                self.gs.goal_selection_vec))
            # save to file
            self.log_position.write(log_string + "\n")
            self.log_position.flush()

        if self.log_predictions is not None :

            # save predictions info on file

            # create log line
            log_string = ""
            # add timing
            log_string += "{:8d} ".format(self.timestep)
            # add predictions info
            curr_predictions = self.gp.w
            for pre in  curr_predictions:
                log_string += "{:6.4f} ".format(pre)
            # add goal index
            log_string += "{:6d}".format(np.argmax(
                self.gs.goal_selection_vec))
            # save to file
            self.log_predictions.write(log_string + "\n")
            self.log_predictions.flush()

        if self.log_targets is not None :

            # save targets info on file

            # create log line
            log_string = ""
            # add timing
            log_string += "{:8d} ".format(self.timestep)
            # add targets info
            keys = sorted(self.gs.target_position.keys())
            all_goals = np.NaN * np.ones([self.gs.N_GOAL_UNITS,
                                         self.gs.N_ROUT_UNITS])

            for key in sorted(self.gs.target_position.keys()):
                all_goals[key, :] = self.gs.target_position[key]

            for angle in all_goals.ravel():
                    log_string += "{:6.4f} ".format(angle)

            # add goal index
            log_string += "{:6d}".format(
                np.argmax(self.gs.goal_selection_vec))
            # save to file
            self.log_targets.write(log_string + "\n")
            self.log_targets.flush()

    def save_weight_logs(self):

        if self.log_weights is not None :
            # create log line
            log_string = ""
            # add timing
            log_string += "{:8d} ".format(self.timestep)
            wm = self.get_weights_metrics()
            log_string += "{:6.4} ".format(wm["kohonen_weights"])
            log_string += "{:6.4} ".format(wm["echo_weights"])

            self.log_weights.write(log_string + "\n")
            self.log_weights.flush()

    def competence_improvement_update(self, im_value, goal_selection_vec):

        # the index of the current highest goal
        win_indx = np.argmax(goal_selection_vec)

        # update the movin' average for that goal
        self.competence_improvement_vec[win_indx] += self.IM_DECAY * (
                -self.competence_improvement_vec[win_indx]
                + self.IM_AMP * im_value)

    def step(self) :

        self.timestep += 1

        self.gm_input = np.zeros([3, self.body_simulator.pixels[0] * 
                                  self.body_simulator.pixels[1] ])

        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            # update the subset of goals to be selected
            self.goal_mask = np.logical_or(self.goal_mask,
                                           (self.gm.goalrep_layer > 0))

            # Selection
            self.gs.goal_selection(self.goal_mask,
                    competence_improvement_vec=
                            self.COMPETENCE_IMPROVEMENT_PROP * 
                            self.competence_improvement_vec,
                    incompetence_vec=(self.INCOMPETENCE_PROP * 
                                         (1.0 - self.gp.w)))

            # Prediction
            if self.gs.goal_window_counter == 0:
                self.gp.step(self.gs.goal_selection_vec)

            # # add vision
            self.gm_input[0] = self.body_simulator.pos_delta.ravel() * 0
            # # add proprioception
            self.gm_input[1] = self.body_simulator.prop_delta.ravel() * 0
            # add touch
            self.gm_input[2] = 5000.0 * self.body_simulator.touch_delta.ravel()

            self.static_inp *= 0
            for inp in self.gm_input :
                self.static_inp += (0.0001 * np.array(inp) / 
                                    float(len(self.gm_input)))
        else:
            if self.gs.reset_window_counter == 0:
                self.static_inp = np.random.rand(*self.static_inp.shape)

        # movement body_simulator step

        self.gs.step(self.static_inp)

        # convert the vector of readout units from the selector into two
        # vectors (left arm , right arm)
        larm_angles = self.gs.out[:(self.gs.N_ROUT_UNITS / 2)]
        rarm_angles = self.gs.out[(self.gs.N_ROUT_UNITS / 2):]
        larm_angles_theoric = self.gs.tout[:(self.gs.N_ROUT_UNITS / 2)]
        rarm_angles_theoric = self.gs.tout[(self.gs.N_ROUT_UNITS / 2):]
        larm_angles_target = self.gs.gout[:(self.gs.N_ROUT_UNITS / 2)]
        rarm_angles_target = self.gs.gout[(self.gs.N_ROUT_UNITS / 2):]

        ########################################################################
        ########################################################################

        if body_simulator_touch_grow == True:
            self.body_simulator.perc.touch_sigma = (
                            body_simulator_touch_sigma * 0.99 * 
                            (1 - self.gp.w.mean()) + 
                            body_simulator_touch_sigma * 0.001)

        # Simulation step
        collision = self.body_simulator.step(
                larm_angles_unscaled=larm_angles,
                rarm_angles_unscaled=rarm_angles,
                larm_angles_theoric_unscaled=larm_angles_theoric,
                rarm_angles_theoric_unscaled=rarm_angles_theoric,
                larm_angles_target_unscaled=larm_angles_target,
                rarm_angles_target_unscaled=rarm_angles_target,
                active=(self.gs.reset_window_counter >= self.gs.RESET_WINDOW)
                )

        if collision == True:
            self.save_cont_logs()
            angles = np.hstack(
                self.body_simulator.actuator.unscale_angles(
                    self.body_simulator.larm_angles,
                    self.body_simulator.rarm_angles))
            self.gs.reset_oscillator(angles, t=self.rng.randint(0,100))
   
            self.current_target = angles.copy()
                

        ########################################################################
        ########################################################################

        if self.gs.reset_window_counter >= self.gs.RESET_WINDOW:

            # Train goal maker
            self.gm.step(self.gm_input,
                         # neigh_scale = self.gp.w.copy())
                         neigh_scale=1 - self.gp.w.mean())
                         # neigh_scale = 1 - self.gp.getCurrPred())

            ####################################################################
            ####################################################################

            # self.gm.learn(eta_scale=(1 - self.gs.getCurrMatch()), pred = (1.0 - self.gp.w) ) # MIXED
            self.gm.learn(eta_scale=(1 - self.gp.getCurrPred()), pred=(1.0 - self.gp.w))  # MIXED-2
            # self.gm.learn(eta_scale=(1 - self.gp.getCurrMatch()), pred = (1.0 - self.gs.match_mean) ) # MIXED-3
            # self.gm.learn(pred = (1.0 - self.gp.w) ) # PRED
            # self.gm.learn(eta_scale=(1 - self.gs.getCurrMatch()) ) # MATCH
            # self.gm.learn(eta_scale=(1 - self.gp.getCurrPred()) ) # MATCH-2

            ####################################################################
            ####################################################################

            # Train experts
            if  self.gs.goal_window_counter > self.gs.GOAL_LEARN_START :
                comp = 1 - gs_eta_decay * self.gp.w.mean()
                self.gs.learn(comp=comp)

            # update counters

            self.gs.goal_window_counter += 1

            # End of trial

            self.match_value = match(
                    self.gm.goalrep_layer,
                    self.gs.goal_selection_vec
                    )

            if (self.match_value == 1 or
                self.gs.goal_window_counter >= self.gs.GOAL_WINDOW) :

                # log matches
                if self.match_value == 1:
                    self.save_match_logs()

                # learn
                self.gp.learn(self.match_value)

                # set current target in case of match
                if self.match_value == 1:
                    angles = np.hstack(
                        self.body_simulator.actuator.unscale_angles(
                            self.body_simulator.larm_angles,
                            self.body_simulator.rarm_angles))
                    self.gs.update_target(angles)

                # log weight metrics
                self.save_weight_logs()

                # update variables
                self.intrinsic_motivation_value = self.gp.prediction_error
                self.competence_improvement_update(
                    self.intrinsic_motivation_value,
                    self.gs.goal_selection_vec)


                self.gs.is_goal_selected = False
                self.gs.reset(match=self.match_value)
                self.body_simulator.reset()

        else:

            # Train goal maker (empty input)
            self.gm.step(self.gm_input)

            self.gs.reset_window_counter += 1
