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
    def __init__(self, n_cells=40*CELLWIDTH**2, activate_thresh=40, potential_thresh=0, weight_thresh=1./2., delta=1./16., initial_weight=7./16.):
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
        #learnable_active_negative_segments = active_segments[~np.isin(active_cells, post)]
        potential_segments = potential_segments[np.isin(potential_cells, unpredicted_active_cells)]
        best_segment_idx = np2.argmaxMulti(potential_overlap[potential_segments], potential_cells)
        learnable_potential_segments = potential_segments[best_segment_idx]
        #print("learnable_positive_segments {}".format(learnable_active_positive_segments))
        #print("learnable_negative_segments {}".format(learnable_active_negative_segments))
        #print("learnable_potential_segments {}".format(learnable_potential_segments))

        self._adjust_weight(learnable_active_positive_segments, pre, self.initial_weight, self.delta*ss, -self.delta*ss)
        #self._adjust_weight(learnable_active_negative_segments, [], self.initial_weight, self.delta*ss, -self.delta*ss)
        self._adjust_weight(learnable_potential_segments, pre, self.initial_weight, self.delta*ss, -self.delta*ss)

        # Step 3 : create new segments and synapses
        new_segment_cells = np.setdiff1d(unpredicted_active_cells, potential_cells)
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
        learnable_potential_segments = potential_segments
        #print("learnable_potential_segments {}".format(learnable_potential_segments))

        self._adjust_weight(learnable_potential_segments, pre, self.initial_weight, self.delta*ss, 0.)


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
    

class GridCellNetwork(object):
    def __init__(self, n_modules=40, n_modals=1,):
        # variables
        self.n_modals = n_modals
        self.L6_cpmodal = n_modules * CELLWIDTH**2
        self.L6_cpmodule = CELLWIDTH**2
        
        # L6
        self.L6 = {mi: [] for mi in range(n_modals)}
        cellCoordinateOffsets = tuple([i * 0.998 + 0.001 for i in range(2)])
        perModRange = 90.0 / float(n_modules)
        for i in range(n_modules):
            orientation = (float(i) * perModRange) + (perModRange / 2.0)
            config = {
                "cellsPerAxis": CELLWIDTH,
                "anchorInputSize": 32*128,
                "scale": float(CELLWIDTH),
                "orientation": np.radians(orientation),
                "activationThreshold": 16,
                "initialPermanence": 1.0,
                "connectedPermanence": 0.5,
                "learningThreshold": 16,
                "sampleSize": -1,
                "permanenceIncrement": 0.1,
                "permanenceDecrement": 0.0,
                "cellCoordinateOffsets": cellCoordinateOffsets,
                "anchoringMethod": "corners"
            }
            for mi in range(n_modals):
                self.L6[mi].append(Superficial2DLocationModule(**config))
        
        # L4
        config = {
            "initialPermanence": 1.0,
            "activationThreshold": int(math.ceil(0.9 * n_modules)),
            "reducedBasalThreshold": int(math.ceil(0.9 * n_modules)),
            "minThreshold": n_modules,
            "sampleSize": n_modules,
            "cellsPerColumn": 32,
            "columnCount": 128,
            "basalInputSize": n_modals*n_modules*CELLWIDTH**2
        }
        self.L4 = ApicalTiebreakPairMemory(**config)
        pass

    
    def movementCompute(self, displacement, mi=0):
        location = {
            "displacement": [displacement["top"],
                             displacement["left"]],
        }
        for module in self.L6[mi]:
            module.movementCompute(**location)
        
        return location
    
    
    def sensoryCompute(self, vec=None, learn=True, mi=0):
        # L4
        basalInput = self.getLR(mi=mi) if learn \
            else [x for mi in range(self.n_modals) for x in self.getLR(mi=mi)]
        l4_input = {
            "activeColumns": vec,
            "basalInput": basalInput,
            "basalGrowthCandidates": self.getLLR(mi=mi),
            "learn": learn
        }
        self.L4.compute(**l4_input)

        # L6
        l6_input = {
            "anchorInput": self.L4.getActiveCells(),
            "anchorGrowthCandidates": self.L4.getWinnerCells(),
            "learn": learn,
        }
        if learn:
            for module in self.L6[mi]:
                module.sensoryCompute(**l6_input)
        else:
            for layer in self.L6.values():
                for module in layer:
                    module.sensoryCompute(**l6_input)
        
        pass
    
    
    def reset(self):
        self.L4.reset()
        for layer in self.L6.values():
            for module in layer:
                module.reset()
        pass


    def activateRandomLocation(self, mi=0, fixed_phases=[]):
        if fixed_phases:
            for iter, module in enumerate(self.L6[mi]):
                module.activateFixedLocation(fixed_phases[iter])
        else:
            active_phases = []
            for module in self.L6[mi]:
                active_phase = module.activateRandomLocation()
                active_phases.append(active_phase)
        
            return active_phases
        pass

    
    def getLR(self, mi=0):
        activeCells = np.array([], dtype="uint32")

        #totalPrevCells = 0
        for i, module in enumerate(self.L6[mi]):
            activeCells = np.append(activeCells, module.getActiveCells() + self.L6_cpmodule * i + self.L6_cpmodal * mi)
            #totalPrevCells += module.numberOfCells()

        return activeCells

    
    def getLLR(self, mi=0):
        learnableCells = np.array([], dtype="uint32")

        #totalPrevCells = 0
        for i, module in enumerate(self.L6[mi]):
            learnableCells = np.append(learnableCells, module.getLearnableCells() + self.L6_cpmodule * i + self.L6_cpmodal * mi)
            #totalPrevCells += module.numberOfCells()

        return learnableCells

    
    def getSALR(self, mi=0):
        cells = np.array([], dtype="uint32")
        #prev_cells = 0
        for i, module in enumerate(self.L6[mi]):
            cells = np.append(cells, module.sensoryAssociatedCells + self.L6_cpmodule * i + self.L6_cpmodal * mi)
            #prev_cells += module.numberOfCells()
        return cells


class ControlGCN(object):
    def __init__(self, n_modals=1, n_modules=40, n_classes=10, T=25, recommendation=False):
        self.network = GridCellNetwork(n_modules=n_modules, n_modals=n_modals)
        self.l6_cells = n_modals * n_modules * CELLWIDTH**2
        self.l6_cpmodal = n_modules * CELLWIDTH**2
        self.network_move = WeightMatrix(n_cells=self.l6_cells, activate_thresh=n_modules)
        self.n_modals = n_modals
        self.n_modules = n_modules
        self.n_classes = n_classes
        self.reccommendation = recommendation
        self.linear = np.zeros((self.l6_cells, n_classes))
        self.current = {mi: None for mi in range(n_modals)}
        self.T = T
        self.ent_mean = [1.05, 0.90, 0.995, 1.26]
        self.ent_var = [0.856, 0.835, 0.763, 0.697]
        pass
    
    
    def learn(self, vecs=None, target=None):
        self.reset()
        fixed_phases = []

        for mi in range(self.n_modals):
            #scatter = myscatter()
            fixed_phases = self.network.activateRandomLocation(mi=mi, fixed_phases=fixed_phases)
            assigned_cells = []
            for t in range(25):
                pi = t
                coordinate = self.pi2coordinate(pi)
                vec = (np.where(vecs[mi][pi] == 1))[0]
                self.move(coordinate, mi=mi)
                self.network.sensoryCompute(vec, learn=True, mi=mi)
                #scatter.add_cell(self.network.getLR(mi=0)[0])

                assigned_cell = self.network.getSALR(mi=mi)
                assigned_cells.extend(assigned_cell)
            
            self.linear[list(set(assigned_cells)), target] += 1
            #scatter.plot()
        
        pass
    
    
    def learn_movement(self, vecs=None, target=None, dpc_idx=0):
        
        # Step0-0. Set mi_T & pi_T
        mi_T = [0]*25 + [1]*25
        pi_T = [i for i in range(25)] + [i for i in range(0, 25)]
        pair = list(zip(pi_T, mi_T))
        random.shuffle(pair)
        pi_T, mi_T = zip(*pair)

        # Step0-1. Initialize Variable
        self.reset()
        z_idx_prev = []
        ent_prev = entropy([1./self.n_classes]*self.n_classes)
        ent_mean = self.ent_mean[dpc_idx]
        ent_var = self.ent_var[dpc_idx]
        ent_seq = np.zeros((len(mi_T, )), dtype=np.float32)
        ss_seq = np.zeros((len(mi_T)), dtype=np.float32)
        
        
        # Step1. Start Learning
        for t in range(len(mi_T)): #self.T
            self.share_l6() if self.n_modals > 1 else None
            
            # Update
            pi = pi_T[t]
            mi = mi_T[t]
            coordinate = self.pi2coordinate(pi)
            vec = (np.where(vecs[mi][pi] == 1))[0]
            self.move(coordinate, mi='all')
            self.network.sensoryCompute(vec, learn=False, mi='all')

            # Evaluation
            z_idx = [x for mi in range(self.n_modals) for x in self.network.getSALR(mi=mi)]
            z = np.zeros(self.l6_cells)
            z[z_idx] = 1
            y = np.matmul(z, self.linear)
            distribution = y / sum(y) if max(y) != 0 else y

            # learn movement
            ent = entropy(distribution) if max(distribution) != 0 else entropy([1./self.n_classes]*self.n_classes)
            ss = (ent_mean - ent) / math.sqrt(ent_var)
            # new
            if ss > 0:
                self.network_move.learn(z_idx_prev, z_idx, ss=ss)
            elif ss < 0:
                self.network_move.learn_negative(z_idx_prev, z_idx, ss=ss)
            
            # old
            """if ent < ent_prev:
                #tqdm.write('good, len(pre) and len(post) {} {}'.format(len(z_idx_prev), len(z_idx)))
                self.network_move.learn(z_idx_prev, z_idx)"""
            
            z_idx_prev = z_idx

        dict = {
            'ent_LM': ent_seq,
            'ss_LM': ss_seq,
            'mi_T_LM': mi_T,
            'pi_T_LM': pi_T,
            'target_LM': target,
        }
        return dict
    
    
    def infer(self, vecs=None, target=None, data_idx=0):
        self.reset()

        pi_T = range(25)
        #mi_T = [0]*20 + [1]*5
        mi_T = [random.randint(0, 1) for _ in range(25)]
        random.shuffle(pi_T)
        #random.shuffle(mi_T)

        pred_flag = False
        pred_step = None
        brief_seq = np.zeros(self.T, dtype=np.float16)
        mf_seq = np.zeros((self.T, self.n_modals), dtype=np.float16)
        ent_seq = np.zeros(self.T, dtype=np.float16)

        for t in range(self.T):
            # 1. Share
            #tqdm.write('start share_l6') if self.n_modals > 1 else None
            self.share_l6(data_idx=data_idx) if self.n_modals > 1 else None

            # 2. Update
            pi = pi_T[t]
            mi = mi_T[t]
            coordinate = self.pi2coordinate(pi)
            vec = (np.where(vecs[mi][pi] == 1))[0]
            #tqdm.write('start move and sensoryCompute')
            self.move(coordinate, mi='all')
            self.network.sensoryCompute(vec, learn=False, mi='all')

            # 3. Evaluation
            z_idx = [x for mi in range(self.n_modals) for x in self.network.getSALR(mi=mi)]
            z = np.zeros(self.l6_cells)
            z[z_idx] = 1
            y = np.matmul(z, self.linear)
            
            # 4.1 new classify
            distribution = y / sum(y) if max(y) != 0 else y
            ent = entropy(distribution) if max(distribution) != 0 else entropy([1./self.n_classes]*self.n_classes)
            ent_seq[t] = ent
            brief_seq[t] = distribution[target]

            # 4.2 old classify
            if not pred_flag and t >= 4 and max(distribution) >= 0.3:
                pred = np.argmax(y)
                pred_flag = True
                if pred == target:
                    pred_step = t
            
            # 5. reccommend
            if self.reccommendation:
                assert self.n_modals <= 2, 'recommendation for more than 2 modals is not implemented'

                #tqdm.write('start network_move.infer')
                z_idx_next = self.network_move.infer(z_idx)
                mf1 = np.sum(z_idx_next < self.l6_cpmodal) if len(z_idx_next) != 0 else 0
                mf2 = np.sum(z_idx_next >= self.l6_cpmodal) if len(z_idx_next) != 0 else 0
                mf = np.array([mf1, mf2])
                if np.all(mf == mf[0]):
                    continue
                else:
                    if t < self.T - 1:
                        mi_T[t+1] = np.argmax(mf)
                mf_seq[t] = mf
            
        # 6. return
        dict = {
            'brief_Eval': brief_seq,
            'pred_step_Eval': pred_step,
            'mf_Eval': mf_seq,
            'mi_T_Eval': mi_T,
            'pi_T_Eval': pi_T,
            'ent_Eval': ent_seq,
            'target_Eval': target,
            }
            
        return dict
    

    def share_l6(self, data_idx=0):
        for module_idx in range(self.n_modules):
            all_phases = np.concatenate([self.network.L6[j][module_idx].getActivePhases() 
                                         for j in range(self.n_modals) ])
            if all_phases.size > 0:
                all_phases = np.unique(all_phases, axis=0)

            for i in range(self.n_modals):
                self.network.L6[i][module_idx].activePhases = all_phases.copy()
        pass
    
    
    def move(self, coordinate=None, mi=0):
        current = {
            "top": coordinate["top"] + coordinate["height"]/2.,
            "left": coordinate["left"] + coordinate["width"]/2.
        }

        if mi == 'all':
            if self.current[0] is not None:
                for mi in range(self.n_modals):
                    displacement = {"top": current['top'] - self.current[mi]['top'],
                                    "left": current['left'] - self.current[mi]['left'],}
                    self.network.movementCompute(displacement, mi=mi)
        elif self.current[mi] is not None:
            displacement = {
                "top": current['top'] - self.current[mi]['top'],
                "left": current['left'] - self.current[mi]['left'],
            }
            self.network.movementCompute(displacement, mi=mi)
        
        if mi == 'all':
            self.current = {mi: current for mi in range(self.n_modals)}
        else:
            self.current[mi] = current

    
    def pi2coordinate(self, pi=None):
        scale = 20
        coordinate = {}
        coordinate['height'] = scale
        coordinate['width'] = scale
        coordinate['top'] = scale * ( pi // 5 )
        coordinate['left'] = scale * ( pi % 5 )
        return coordinate
    
    
    def reset(self):
        self.network.reset()
        self.current = {mi: None for mi in range(self.n_modals)}
        pass


class myscatter:
    def __init__(self):
        self.x = []
        self.y = []


    def add_cell(self, c):
        self.x.append(c % CELLWIDTH)
        self.y.append(c // CELLWIDTH)
        pass


    def plot(self, name='noname.png'):
        plt.scatter(self.x, self.y)
        plt.xlim((0,255))
        plt.ylim((0,255))
        plt.savefig('results/' + name)
