import numpy as np

simulation_im_amp = 1.0
simulation_im_decay = 0.2
simulation_competence_improvement_prop = 1.0
simulation_incompetence_prop = 0.0


body_simulator_pixels = [20, 20]
body_simulator_lims = [[-5, 5], [-2, 4.]]
body_simulator_touch_th = 0.1
body_simulator_num_touch_sensors = 30
body_simulator_touch_sigma = 0.1
body_simulator_actuator_NUMBER_OF_JOINTS = 3

pm_proprioceptive_retina_sigma = 0.1
pm_touch_retina_sigma = 0.1
pm_touch_retina_sigma_width_scale = 0.2
pm_proprioceptive_retina_sigma_width_scale = 0.2
pm_proprioceptive_angle_threshold = 1e-5
pm_image_resolution = 12

ka_num_joints = body_simulator_actuator_NUMBER_OF_JOINTS
ka_left_origin = [-1.5, 0.0]
ka_left_lims = [
        [0, np.pi],    # first joint limits
        [0, np.pi*0.6],    # second joint limits
        [0, np.pi*0.6]     # third joint limits
        ]
assert(len(ka_left_lims) == body_simulator_actuator_NUMBER_OF_JOINTS )

ka_right_origin = [1.5, 0.0]
ka_right_lims = [
        [0, np.pi],    # first joint limits
        [0, np.pi*0.6],    # second joint limits
        [0, np.pi*0.6]     # third joint limits
        ]
assert(len(ka_right_lims) == body_simulator_actuator_NUMBER_OF_JOINTS )


GOAL_NUMBER = 25

gs_dt = 0.001
gs_tau = 0.04
gs_alpha = 0.04
gs_epsilon = 1.0e-10
gs_eta = 0.06
gs_n_input = body_simulator_pixels[0]*body_simulator_pixels[1]
gs_n_goal_units = GOAL_NUMBER
gs_n_echo_units = 800
gs_n_rout_units = body_simulator_actuator_NUMBER_OF_JOINTS*2
gs_match_decay = 0.2
gs_noise = 1.0
gs_noise_scale = 30.0
gs_sm_temp = 0.01
gs_g2e_spars = 0.01
gs_echo_ampl = 5.0
gs_goal_window = 100
gs_goal_learn_start = 20
gs_reset_window = 10
gs_multiple_echo = False


gp_eta = 0.2

robot_stime = 10000

gm_n_input_layers = [gs_n_input, gs_n_input, gs_n_input]
gm_n_singlemod_layers = [64, 64, 64]
gm_n_hidden_layers = [16, 16]
gm_n_out = 16
gm_n_goalrep = GOAL_NUMBER
gm_singlemod_lrs = [0.05, 0.01, 0.1]
gm_hidden_lrs = [0.001, 0.001]
gm_output_lr = 0.001
gm_goalrep_lr = 0.8
gm_goal_th = 0.1
gm_stime = robot_stime
gm_single_kohonen = True

gm_singlemod_n_dim_out = 2
gm_singlemod_eta_bl_scale = 4.0
gm_singlemod_eta_decay_scale = 4.0
gm_singlemod_neigh_scale = 2.0
gm_singlemod_neigh_decay_scale = 4.0
gm_singlemod_neigh_bl = 0.5
gm_singlemod_weight_bl = 0.001

gm_hidden_n_dim_out = 2
gm_hidden_eta_bl_scale = 4.0
gm_hidden_eta_decay_scale = 4.0
gm_hidden_neigh_scale = 2.0
gm_hidden_neigh_decay_scale = 4.0
gm_hidden_neigh_bl = 0.5
gm_hidden_weight_bl = 0.001

gm_out_n_dim_out = 2
gm_out_eta_bl_scale = 4.0
gm_out_eta_decay_scale = 4.0
gm_out_neigh_scale = 2.0
gm_out_neigh_decay_scale = 4.0
gm_out_neigh_bl = 0.5
gm_out_weight_bl = 0.001

gm_single_kohonen_n_dim_out = 2
gm_single_kohonen_eta_bl = 0.0
gm_single_kohonen_eta_decay = 1e20
gm_single_kohonen_neigh_scale = 0.01
gm_single_kohonen_neigh_decay_scale = 1e20
gm_single_kohonen_neigh_bl = 0.0
gm_single_kohonen_weight_bl = 0.001

gm_goalrep_n_dim_out = 1
gm_goalrep_eta_bl_scale = 4.0
gm_goalrep_eta_decay_scale = 4.0
gm_goalrep_neigh_scale = 0.2
gm_goalrep_neigh_decay_scale = 1.0e400
gm_goalrep_neigh_bl = 0.5
gm_goalrep_weight_bl = 0.001
