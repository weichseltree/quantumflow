import os
import pandas as pd
import numpy as np
import tensorflow as tf

import quantumflow

def calculate_potentials(a, b, c, 
                         discretisation_points=500, 
                         interval_length=1.0,
                         dtype='float64',
                         **kwargs):
    
        a = tf.cast(a, dtype)
        b = tf.cast(b, dtype)
        c = tf.cast(c, dtype)

        x = tf.linspace(tf.constant(0.0, dtype=dtype), interval_length, discretisation_points)

        curves = -tf.square(tf.expand_dims(tf.expand_dims(x, 0), 2) - b)/(2*tf.square(c))
        curves = -a*tf.exp(curves)

        potential = tf.reduce_sum(curves, -1)
        h = tf.cast(interval_length/(discretisation_points-1), dtype=dtype) # discretisation interval

        return potential, x, h

class TXTPotentialDataset(quantumflow.QFDataset):

    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.kwargs = kwargs

    def build(self):
        paper_coeff = pd.read_csv(os.path.join(self.run_dir, self.filename), delimiter=' ')

        a = paper_coeff[['a1', 'a2', 'a3']].values[:, np.newaxis, :]
        c = paper_coeff[['b1', 'b2', 'b3']].values[:, np.newaxis, :] # b<->c listed wrong in the paper
        b = paper_coeff[['c1', 'c2', 'c3']].values[:, np.newaxis, :]

        self.potential, self.x, self.h = calculate_potentials(a, b, c, **self.kwargs)