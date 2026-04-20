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
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from A_ApicalTemporalMemory import ApicalTiebreakPairMemory
from A_LocationModule import Superficial2DLocationModule
from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections

CELLWIDTH = 10

class GridCellNetwork(object):
    def __init__(self, n_modules=10, n_modals=1,):
        # variables
        self.n_modals = n_modals
        self.L6_cpmodal = n_modules * CELLWIDTH**2
        self.L6_cpmodule = CELLWIDTH**2
        np.random.seed(42)
        random.seed(42)
        
        # L6
        self.L6 = {mi: [] for mi in range(n_modals)}
        cellCoordinateOffsets = tuple([i * 0.998 + 0.001 for i in range(2)])
        perModRange = 90.0 / float(n_modules)
        for i in range(n_modules):
            orientation = (float(i) * perModRange) + (perModRange / 2.0)
            config = {
                "cellsPerAxis": CELLWIDTH,
                "anchorInputSize": 4*16,
                "scale": float(CELLWIDTH),
                "orientation": np.radians(orientation),
                "activationThreshold": 4,
                "initialPermanence": 1.0,
                "connectedPermanence": 0.5,
                "learningThreshold": 4,
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
            "cellsPerColumn": 4,
            "columnCount": 16,
            "basalInputSize": n_modals*n_modules*CELLWIDTH**2
        }
        self.L4 = ApicalTiebreakPairMemory(**config)
        pass
    
    
    def movementCompute(self, displacement, mi=0):
        location = {
            "displacement": [-displacement["top"],
                             displacement["left"]],
        }
        for module in self.L6[mi]:
            module.movementCompute(**location)
        
        return location
    
    
    def sensoryCompute(self, vec=None, learn=True, mi=0):
        # L4
        basalInput = self.getLR(mi=mi) if learn \
            else [x for i in range(self.n_modals) for x in self.getLR(mi=i)]
        basal_GC = self.getLLR(mi=mi) if learn else None
        l4_input = {
            "activeColumns": vec,
            "basalInput": basalInput,
            "basalGrowthCandidates": basal_GC,
            "learn": learn,
        }
        self.L4.compute(**l4_input)
        #print('l4 input {}'.format(l4_input))
        #print('l4 active cells {}'.format(self.L4.getActiveCells()))
        print('l4 basal input {}'.format(basalInput))

        # L6
        anchorinput = self.L4.getActiveCells()
        #anchorinput = [0,5,10,12] if not learn else anchorinput
        l6_input = {
            "anchorInput": anchorinput,
            "anchorGrowthCandidates": self.L4.getWinnerCells(),
            "learn": learn,
        }
        #print('l6 input {}'.format(l6_input))
        print('l4 active cells {}'.format(self.L4.getActiveCells()))
        print('l4 growth cand {}'.format(self.L4.getWinnerCells()))
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


    def activateRandomLocation(self, mi=0, phases=[]):
        if phases:
            for iter, module in enumerate(self.L6[mi]):
                module.activateFixedLocation(phases[iter])
            return phases
        else:
            active_phases = [module.activateRandomLocation() for module in self.L6[mi]]        
            return active_phases

    
    def getLR(self, mi=0):
        activeCells = np.array([], dtype="uint32")
        for i, module in enumerate(self.L6[mi]):
            activeCells = np.append(activeCells, module.getActiveCells() + self.L6_cpmodule * i + self.L6_cpmodal * mi)

        return activeCells

    
    def getLLR(self, mi=0):
        learnableCells = np.array([], dtype="uint32")

        for i, module in enumerate(self.L6[mi]):
            learnableCells = np.append(learnableCells, module.getLearnableCells() + self.L6_cpmodule * i + self.L6_cpmodal * mi)

        return learnableCells

    
    def getSALR(self, mi=0):
        cells = np.array([], dtype="uint32")
        for i, module in enumerate(self.L6[mi]):
            cells = np.append(cells, module.sensoryAssociatedCells + self.L6_cpmodule * i + self.L6_cpmodal * mi)
        return cells
    

    def getSALR_1module(self, mi=0):
        cells = np.array([], dtype="uint32")
        for i, module in enumerate(self.L6[mi]):
            cells = np.append(cells, module.sensoryAssociatedCells + self.L6_cpmodule * i + self.L6_cpmodal * mi)
            break
        return cells


if __name__ == "__main__":
    gcn = GridCellNetwork()

    # Learn
    print('----- Learn -----')
    gcn.activateRandomLocation()
    gcn.sensoryCompute(vec=[0,1,2,3,], learn=True)
    gcn.reset()

    gcn.activateRandomLocation()
    gcn.sensoryCompute(vec=[0,1,2,3,], learn=True)
    gcn.reset()
    # Eval
    print('----- Eval -----')
    gcn.sensoryCompute(vec=[0,1,2,3,], learn=False)
    print('l6 pred: {}'.format(gcn.getSALR(mi=0)))

    gcn.sensoryCompute(vec=[4,1,2,3,], learn=False)
    print('l6 pred: {}'.format(gcn.getSALR(mi=0)))


