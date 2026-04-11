# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Emulates a grid cell module"""

import math
import random
import copy
import numpy as np
from tqdm import tqdm
from htmresearch.support import numpy_helpers as np2
from htmresearch.algorithms.multiconnections import Multiconnections
from nupic.bindings.math import SparseMatrixConnections, Random


class Superficial2DLocationModule(object):
  def __init__(self,
               cellsPerAxis,
               scale,
               orientation,
               anchorInputSize,
               cellCoordinateOffsets=(0.5,),
               activationThreshold=10,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               learningThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.0,
               maxSynapsesPerSegment=-1,
               anchoringMethod="narrowing",
               rotationMatrix = None,
               seed=42):

    self.cellsPerAxis = cellsPerAxis
    self.cellDimensions = np.asarray([cellsPerAxis, cellsPerAxis], dtype="int")
    self.scale = scale
    self.moduleMapDimensions = np.asarray([scale, scale], dtype="float")
    self.phasesPerUnitDistance = 1.0 / self.moduleMapDimensions

    if rotationMatrix is None:
      self.orientation = orientation
      self.rotationMatrix = np.array(
        [[math.cos(orientation), -math.sin(orientation)],
         [math.sin(orientation), math.cos(orientation)]])
      if anchoringMethod == "discrete":
        # Need to convert matrix to have integer values
        nonzeros = self.rotationMatrix[np.where(np.abs(self.rotationMatrix)>0)]
        smallestValue = np.amin(nonzeros)
        self.rotationMatrix /= smallestValue
        self.rotationMatrix = np.ceil(self.rotationMatrix)
    else:
      self.rotationMatrix = rotationMatrix

    self.cellCoordinateOffsets = cellCoordinateOffsets

    # Phase is measured as a number in the range [0.0, 1.0)
    self.activePhases = np.empty((0,2), dtype="float")
    self.cellsForActivePhases = np.empty(0, dtype="int")
    self.phaseDisplacement = np.empty((0,2), dtype="float")
    self.activeCells = np.empty(0, dtype="int")

    # The cells that were activated by sensory input in an inference timestep,
    # or cells that were associated with sensory input in a learning timestep.
    self.sensoryAssociatedCells = np.empty(0, dtype="int")
    self.activeSegments = np.empty(0, dtype="uint32")
    self.connections = SparseMatrixConnections(np.prod(self.cellDimensions),
                                               anchorInputSize)

    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.learningThreshold = learningThreshold
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.anchoringMethod = anchoringMethod
    self.rng = Random(seed)


  def _computeActiveCells(self):
    # Round each coordinate to the nearest cell.
    activeCellCoordinates = np.floor(
      self.activePhases * self.cellDimensions).astype("int")

    # Convert coordinates to cell numbers.
    self.cellsForActivePhases = (
      np.ravel_multi_index(activeCellCoordinates.T, self.cellDimensions))
    self.activeCells = np.unique(self.cellsForActivePhases)


  def activateRandomLocation(self):
    """
    Set the location to a random point.
    """
    self.activePhases = np.array([np.random.random(2)])
    self._computeActiveCells()
    return self.activePhases
  

  def activateFixedLocation(self, phase):
    self.activePhases = phase
    self._computeActiveCells()


  def movementCompute(self, displacement, noiseFactor = 0):
    """
    Shift the current active cells by a vector.

    @param displacement (pair of floats)
    A translation vector [di, dj].
    """


    # Calculate delta in the module's coordinates.
    phaseDisplacement = (np.matmul(self.rotationMatrix, displacement) *
                         self.phasesPerUnitDistance)

    # Shift the active coordinates.
    np.add(self.activePhases, phaseDisplacement, out=self.activePhases)

    # In Python, (x % 1.0) can return 1.0 because of floating point goofiness.
    # Generally this doesn't cause problems, it's just confusing when you're
    # debugging.
    np.round(self.activePhases, decimals=9, out=self.activePhases)
    np.mod(self.activePhases, 1.0, out=self.activePhases)

    self._computeActiveCells()
    self.phaseDisplacement = phaseDisplacement


  def _sensoryComputeInferenceMode(self, anchorInput):
    """
    Infer the location from sensory input. Activate any cells with enough active
    synapses to this sensory input. Deactivate all other cells.

    @param anchorInput (numpy array)
    A sensory input. This will often come from a feature-location pair layer.
    """
    if len(anchorInput) == 0:
      return

    # overlap and active segments and cells
    overlaps = self.connections.computeActivity(anchorInput, self.connectedPermanence)
    activeSegments = np.where(overlaps >= self.activationThreshold)[0]
    sensorySupportedCells = np.unique(self.connections.mapSegmentsToCells(activeSegments))

    inactivated = np.setdiff1d(self.activeCells, sensorySupportedCells)
    inactivatedIndices = np.in1d(self.cellsForActivePhases, inactivated).nonzero()[0]
    if inactivatedIndices.size > 0:
      self.activePhases = np.delete(self.activePhases, inactivatedIndices, axis=0)


    # specialized in Grid Cell
    activatedCoordsBase = np.transpose(np.unravel_index(sensorySupportedCells, self.cellDimensions)).astype('float')
    activatedCoords = np.concatenate(
      [activatedCoordsBase + [iOffset, jOffset]
       for iOffset in self.cellCoordinateOffsets
       for jOffset in self.cellCoordinateOffsets]
    )
    self.activePhases = activatedCoords / self.cellDimensions
    #print(activatedCoords)

    self._computeActiveCells()
    self.activeSegments = activeSegments
    self.sensoryAssociatedCells = sensorySupportedCells

  # most important
  def _sensoryComputeLearningMode(self, anchorInput):
    """
    Associate this location with a sensory input. Subsequently, anchorInput will
    activate the current location during anchor().

    @param anchorInput (numpy array)
    A sensory input. This will often come from a feature-location pair layer.

    self.activeCells is the predicted cells(by location movement), 
    cellsforactive is the predict from L4 input,
    anchorInput is the L4 input
    """
    # Step 1 : activate segments and cells from L4 input
    overlaps = self.connections.computeActivity(anchorInput, self.connectedPermanence)
    activeSegments = np.where(overlaps >= self.activationThreshold)[0]
    cellsForActiveSegments = self.connections.mapSegmentsToCells(activeSegments)

    potentialOverlaps = self.connections.computeActivity(anchorInput)
    matchingSegments = np.where(potentialOverlaps >= self.learningThreshold)[0]
    remainingCells = np.setdiff1d(self.activeCells, cellsForActiveSegments)
    candidateSegments = self.connections.filterSegmentsByCell(matchingSegments, remainingCells)
    cellsForCandidateSegments = (self.connections.mapSegmentsToCells(candidateSegments))
 
    # Step 2 : adjust weight (this doesnt create new synapses)    
    learningActiveSegments = activeSegments[np.in1d(cellsForActiveSegments, self.activeCells)]
    
    # Remaining cells with a matching segment: reinforce the best matching segment.
    candidateSegments = candidateSegments[np.in1d(cellsForCandidateSegments, remainingCells)]
    onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments], cellsForCandidateSegments)
    learningMatchingSegments = candidateSegments[onePerCellFilter]


    for learningSegments in (learningActiveSegments, learningMatchingSegments):
        self._learn(self.connections, self.rng, learningSegments,
                    anchorInput, potentialOverlaps,
                    self.initialPermanence, self.sampleSize,
                    self.permanenceIncrement, self.permanenceDecrement,
                    self.maxSynapsesPerSegment)

    # Step 3 : create new segments and synapses
    # Remaining cells without a matching segment: grow one.
    numNewSynapses = len(anchorInput)
    newSegmentCells = np.setdiff1d(remainingCells, cellsForCandidateSegments)
    newSegments = self.connections.createSegments(newSegmentCells)

    self.connections.growSynapsesToSample(
      newSegments, anchorInput, numNewSynapses,
      self.initialPermanence, self.rng)
    self.activeSegments = activeSegments
    self.sensoryAssociatedCells = self.activeCells

  # not important
  def sensoryCompute(self, anchorInput, anchorGrowthCandidates, learn):
    if learn:
      self._sensoryComputeLearningMode(anchorGrowthCandidates)
    else:
      self._sensoryComputeInferenceMode(anchorInput)


  @staticmethod
  def _learn(connections, rng, learningSegments, activeInput,
             potentialOverlaps, initialPermanence, sampleSize,
             permanenceIncrement, permanenceDecrement, maxSynapsesPerSegment):
    """
    Adjust synapse permanences, grow new synapses, and grow new segments.

    @param learningActiveSegments (numpy array)
    @param learningMatchingSegments (numpy array)
    @param segmentsToPunish (numpy array)
    @param activeInput (numpy array)
    @param potentialOverlaps (numpy array)
    """
    # Learn on existing segments
    connections.adjustSynapses(learningSegments, activeInput, permanenceIncrement, -permanenceDecrement)
    maxNew = len(activeInput)
    connections.growSynapsesToSample(learningSegments, activeInput, maxNew, initialPermanence, rng)


  def getActiveCells(self):
    return self.activeCells
  

  def getActivePhases(self):
    return self.activePhases


  def getLearnableCells(self):
    return self.activeCells


  def getSensoryAssociatedCells(self):
    return self.sensoryAssociatedCells


  def addActivePhases(self, phases):
    if phases.size == 0:
      return
    
    phases = phases.reshape(-1, 2)
    self.activePhases = np.concatenate([self.activePhases, phases])
    #print(self.activePhases)
    #self.activePhases = np.unique(self.activePhases)
    #if len(self.activePhases) > 1e7:
    #tqdm.write('len(actPhases) {}'.format(len(self.activePhases)))
      #exit()
    pass


  def reset(self):
    """
    Clear the active cells.
    """
    self.activePhases = np.empty((0,2), dtype="float")
    self.phaseDisplacement = np.empty((0,2), dtype="float")
    self.cellsForActivePhases = np.empty(0, dtype="int")
    self.activeCells = np.empty(0, dtype="int")
    self.sensoryAssociatedCells = np.empty(0, dtype="int")
    pass

  
  def numberOfCells(self):
    return np.prod(self.cellDimensions)
