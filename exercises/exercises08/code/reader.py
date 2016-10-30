import numpy as np
import scipy as sp
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve as sparse_solve
import random

def read_file(fname):
    """Read csv from fname, returning a matrix"""

    datamat = np.loadtxt(fname,delimiter=",",skiprows=1)
    sz = np.size(datamat)
    return np.reshape(datamat,(sz, 1),order='C')

def gen_edgeNeighbors(gridEdgeSize):
    """Generate the edgelist and appropriate neighbors

    Assuming that the vertices are in a gridEdgeSize x gridEdgeSize grid and
    numbered in row-major order and zero-indexed. Generate a list of edges
    and their corresponding endpoints as an ordered tuple (j,k) where j < k

    Example: if gridEdgeSize is 4, we have the following node numbering scheme:

    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15

    The edges are generated in the order shown:

    0-- 0 --1--  2 --2-- 4  --3
    |       |        |        |
    1       3        5        6
    |       |        |        |
    4-- 7 --5--  9 --6-- 11 --7
    |       |        |        |
    8      10       12       13
    |       |        |        |
    8-- 14--9-- 16--10-- 18--11
    |       |        |        |
    15      17      19       20
    |       |        |        | ff
    12--21--13--22--14-- 23--15

    The nodes are visited in row-major order, and an attempt is made to
    generate two edges, leading right and leading down.
    If these edges are not possible, they are skipped.

    We do not explicitly refer to edge numbers in this function, instead a pair of
    tuples is generated in the order specified above, so the output for the above
    grid would begin (0,1), (1,2), (2,3), (4,5), ...
    """
    # Number of nodes = number of cells = n^2
    numNodes = gridEdgeSize * gridEdgeSize
    maxValidNodeId = numNodes - 1

    ## Define utility functions to help in the main generation: are these IDs
    # at the right side or along the bottom of the square grid?
    def isRightSide(n):
        return (n + 1) % gridEdgeSize == 0

    def isBottomSide(n):
        return n >= numNodes - gridEdgeSize

    nodeId = 0
    # First generate all horizontal edges by iterating through all possible
    # left endpoints and generating an edge where appropriate
    while nodeId < numNodes:
        ## Try to generate an edge extending to the right
        if isRightSide(nodeId):
            pass # No possible edge extends to the right--skip
        else:
            yield (nodeId, nodeId + 1)

        ## Try to generate an edge extending down
        if isBottomSide(nodeId):
            pass #No edge extending down
        else:
            yield(nodeId, nodeId + gridEdgeSize)

        nodeId += 1

def gen_D_matrix(gridEdgeSize):
    """Generate the oriented edge matrix of a graph for a size x size grid"""

    # Number of nodes = number of cells = n^2
    numNodes = gridEdgeSize * gridEdgeSize
    maxValidNodeId = numNodes - 1

    # Calculate the total number of edges by building up 3 different classes of nodes
    # Center nodes have 4 edges, edge nodes have 3 edges, corner nodes have 2 edges
    # This will count each edge twice, so divide by 2 at the end
    e1 = gridEdgeSize - 2
    numEdges = e1 * e1 * 4 # There are e1 * e1 nodes in the interior, each has 4 edges
    numEdges += e1 * 4 * 3 # There are e1 nodes on the boundary, each one has 3 edges
    numEdges += 4 * 2      # There are 4 corner nodes, each has 2 edges
    numEdges //= 2         # Account for double-counting

    # Use a linklist to build the matrix, then convert to CSC at the end
    D = spsp.lil_matrix((numEdges,numNodes))

    edgeIndex = 0
    for (a,b) in gen_edgeNeighbors(gridEdgeSize):
        D[edgeIndex, a] = 1
        D[edgeIndex, b] = -1
        edgeIndex += 1

    return spsp.csc_matrix(D)
