import numpy as np
import matplotlib.pyplot as plt

class Oscillator(object):
    def __init__(self, scale, freqs, res = None):
        '''
        :param scale: scaling factor for the frequency
        :type scale: float

        :param freqs: list of frequencies (one for each trjectory)
        :type freqs: list(float)
        '''

        self.scale = scale
        self.freqs = np.array(freqs)
        self.trajectories_num = self.freqs.size  
        
        self.t = np.random.randint(0,40)

        self.res = 0 if res is None else res

    def __call__(self):
        """ Calls an oscillator's step

        :return: the current angles after one step
        :rtype: numpy.array(trajectories_num, dtype=float)
        """
        
        self.res += 0.001*(-self.res -(0.006**-1)*np.sin(2.0 * np.pi * self.freqs * self.t / 
                       self.scale + np.pi)) * 0.5
        self.res = np.tanh(self.res)
        self.t += 1

        return self.res

    def reset(self):

        self.t = 0

plt.close("all")

stime = 100
f_n = 6

o = Oscillator(6, np.random.uniform(-1,1,f_n), np.random.uniform(-1,1,f_n))

y = np.zeros([f_n,stime])
for t in range(stime):
    y[:, t] = o()

plt.plot(y.T)
plt.show()
