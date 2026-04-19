#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import math
import numpy as np
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections
from A_ApicalTemporalMemory import ApicalTiebreakPairMemory
from A_LocationModule import Superficial2DLocationModule
CELLWIDTH = 256


class WeightMatrix(object):
    def __init__(self, n_cells=40*CELLWIDTH**2, activate_thresh=40, potential_thresh=0, weight_thresh=1./2., delta=0.1, initial_weight=0.4):
        self.weight = SparseMatrixConnections(n_cells, n_cells)
        self.pre_active_cells = []
        self.active_cells = []
        self.active_threshold = activate_thresh
        self.potential_threshold = potential_thresh
        self.weight_threshold = weight_thresh
        self.delta = delta
        self.initial_weight = initial_weight
        self.rng = Random(42)


    def learn(self, pre, post, ss=1.):

        # Step 1 : activate segments and cells
        overlap = self.weight.computeActivity(pre, self.weight_threshold)
        active_segments = np.where(overlap >= self.active_threshold)[0]
        active_cells = self.weight.mapSegmentsToCells(active_segments)
        unpredicted_active_cells = np.setdiff1d(post, active_cells)

        potential_overlap = self.weight.computeActivity(pre)
        potential_segments = np.where(potential_overlap >= self.potential_threshold)[0]
        potential_segments = self.weight.filterSegmentsByCell(potential_segments, unpredicted_active_cells)
        potential_cells = self.weight.mapSegmentsToCells(potential_segments)
        #print("active_segments {}".format(active_segments))
        #print("potential_segments {}".format(potential_segments))


        # Step 2 : adjust wieght (this doesnt create new synapses)
        learnable_active_positive_segments = active_segments[np.isin(active_cells, post)]
        learnable_active_negative_segments = active_segments[~np.isin(active_cells, post)]
        potential_segments = potential_segments[np.isin(potential_cells, unpredicted_active_cells)]
        best_segment_idx = np2.argmaxMulti(potential_overlap[potential_segments], potential_cells)
        learnable_potential_segments = potential_segments[best_segment_idx]
        print("learnable_positive_segments {}".format(learnable_active_positive_segments))
        print("learnable_negative_segments {}".format(learnable_active_negative_segments))
        print("learnable_potential_segments {}".format(learnable_potential_segments))

        self._adjust_weight(learnable_active_positive_segments, pre, self.initial_weight, self.delta*ss, -self.delta*ss)
        #self._adjust_weight(learnable_active_negative_segments, [], self.initial_weight, self.delta*ss, -self.delta*ss)
        self._adjust_weight(learnable_potential_segments, pre, self.initial_weight, self.delta*ss, -self.delta*ss)

        # Step 3 : create new segments and synapses
        new_segment_cells = np.setdiff1d(unpredicted_active_cells, potential_cells)
        #print('new_segment_cells {}'.format(new_segment_cells))
        newSegments = self.weight.createSegments(new_segment_cells)
        n_newsynapses = len(pre)
        self.weight.growSynapsesToSample(newSegments, pre, n_newsynapses, self.initial_weight, self.rng)
        self.pre_active_cells = active_cells
        pass


    def learn_negative(self, pre, post, ss=-1.):
        
        # Step 1 : activate segments and cells
        potential_overlap = self.weight.computeActivity(pre)
        potential_segments = np.where(potential_overlap >= self.potential_threshold)[0]
        potential_segments = self.weight.filterSegmentsByCell(potential_segments, post)
        #print("potential_segments {}".format(potential_segments))


        # Step 2 : adjust wieght (this doesnt create new synapses)
        learnable_negative_segments = potential_segments
        print("learnable_negative_segments {}".format(learnable_negative_segments))

        self._adjust_weight(learnable_negative_segments, pre, self.initial_weight, self.delta*ss, 0)


    def _adjust_weight(self, learnable_segments, input, initial, inc, dec):
        self.weight.adjustSynapses(learnable_segments, input, inc, dec)


    def _grow_synapses(self, learnable_segments, input, initial):
        n_new_synapses = len(input)
        self.weight.growSynapsesToSample(learnable_segments, input, n_new_synapses, initial, self.rng)


    def infer(self, input):
        if len(input) == 0:
            return []
        
        overlap = self.weight.computeActivity(input, self.weight_threshold)
        active_segments = np.where(overlap >= self.active_threshold)[0]
        active_cells = np.unique(self.weight.mapSegmentsToCells(active_segments))

        return active_cells


if __name__ == "__main__":
    w = WeightMatrix(activate_thresh=3)
    
    # Learn
    ss = 1.
    for i in range(2):
        pre = [0,1,2,3,4]
        post = [10,11,12]
        w.learn(pre, post, ss)
        
    print("===========")
    ss = -1.
    for i in range(1):
        pre = [0,1,]
        post = [10,11,12]
        #w.learn_negative(pre, post, ss)
    print("===========")

    # Infer
    pre = [0,1,2,3,4]
    pred = w.infer(pre)
    print("pre = {} \npred = {}".format(pre, pred))

    pre = [0,1,2,]
    pred = w.infer(pre)
    print("pre = {} \npred = {}".format(pre, pred))

    i = 5
    for i in range(3):
        print(i)
    print(i)
