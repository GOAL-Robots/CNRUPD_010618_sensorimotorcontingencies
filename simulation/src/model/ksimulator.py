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
import kinematics as K
np.set_printoptions(suppress=True, precision=5, linewidth=9999999)


--------------------------------------------------------------------


class simulator(object):
    """ Simulate collisions between moving arms (joined tokens)
            and objects (simple tokens)
    """

    def __init__(self):

        # a dictionary of objects
        self.objects = dict()
        self.arms = dict()
        pass

    def add_arm(self, arm):
        pass

    def delete_arm(self, arm):
        pass

    def get_arm(self, name):
        """
        :param  name:  name of the current arm
        """
        pass

    def add_object(self, object):
        pass

    def delete_object(self, object):
        pass

    def get_object(self, name):
        """
        :param  name:  name of the current object
        """
        pass

    def step(self, inputs):
        pass


#------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
