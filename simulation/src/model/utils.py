import numpy as np


def hist_comp(x):
    h = np.histogram(x) 
    freqs = h[0]
    centr = np.linspace(h[1][0],h[1][-1], 
            2*len(h[1])-1 )[1::2]
    return np.vstack([freqs, centr]).T


#################################################################
#################################################################

def reshape_weights(w):
    reshaped_w = []
    reshaped_w_raw = []
    single_w_raws = int(np.sqrt(len(w[0])))
    single_w_cols = single_w_raws

    n_single_w = len(w)
    out_raws = np.sqrt(n_single_w)
    for single_w, i in zip(w, xrange(n_single_w)):
        reshaped_w_raw.append(
            single_w.reshape(single_w_cols,
                single_w_raws, order="F"))
        if (i+1)%out_raws == 0:
            reshaped_w.append(np.hstack(reshaped_w_raw))
            reshaped_w_raw =[]
    reshaped_w = np.vstack(reshaped_w)

    return reshaped_w

def reshape_coupled_weights(w):
    '''
    Reshape the matrix of weight from two (nxn) input layers
    to one (mxm) output layers so that it becomes a block matrix
    made by  mxm  nx2n matrices
    '''

    reshaped_w = []
    reshaped_w_raw = []
    single_w_raws = int(np.sqrt(len(w[0])/2))
    single_w_cols = 2*single_w_raws

    n_single_w = len(w)
    out_raws = np.sqrt(n_single_w)
    for single_w, i in zip(w, xrange(n_single_w)):
        reshaped_w_raw.append(
            single_w.reshape(single_w_cols,
                single_w_raws, order="F"))
        if (i+1)%out_raws == 0:
            reshaped_w.append(np.hstack(reshaped_w_raw))
            reshaped_w_raw =[]
    reshaped_w = np.vstack(reshaped_w)

    return reshaped_w

def angles2positions(angles_array, act = None):
    
    if act is None:
        act = KinematicActuator()
    
    positions = []
    num_angles = len(angles_array[0])
    for angles in angles_array:
        l_angles = angles[:(num_angles/2)]
        r_angles = angles[(num_angles/2):]
        l_angles, r_angles = act.rescale_angles(l_angles, r_angles)
        act.set_positions_basedon_angles(l_angles, r_angles)
    
        positions.append(np.vstack((act.position_l[::-1], act.position_r)))
    
    return positions
        