# DATA PROCESSING LIBRARY FOR PROCESSING THE SEAN LIDDICK DATA
##############################################################
#                                                            #
#  AUTHOR: HARRISON LABOLLITA                                #
#  DATE: JULY 2018                                           #
#  VERSION: 1.0                                              #
#                                                            #
##############################################################
# These functions are used throughout the python scripts in this repository
# This file is imported in almost all of them to import the data provided by
# Sean Liddick.
# ------------------------------------------------------------------------------
def get_data(filename):
    import csv
    import numpy as np

    """
    Expects file "BetaScint2DEnergy.csv" or BetaScint2DEnergy2Electron.csv"

    INPUTS: csv file of Sean Liddick data

    OUTPUTS: The inputs for the machine learning algorithm, a grid of the detector or a vector depending on the
    need, adjust accordingly. And a vector of outputs containing the initial energy and origin of electron
    """

    file = open(filename)
    events = csv.reader(file)
    data = []
    for event in events:
        data.append(event)
    data = np.asarray(data)
    N, M = data.shape
    val = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            val[i][j] = float(data[i][j])
    grid = val[:,0:256]
    outs = val[:,256:]
    events = []
    outputs = []
    for i in range(len(grid)):
        events.append(np.transpose(grid[i,:].reshape((16,16)))) #We need to flip the matrix to match the correct spatial orientation
    for i in range(len(outs)):
        outputs.append(outs[i,:])
    return events, outputs
# ------------------------------------------------------------------------------
def max_energy_matrix(grid):
    """
    This function finds the maximum energy deposited in a single pixel i,j

    INPUT: Matrix representing the detector grid.

    OUTPUT: Maximum energy deposited in single pixel on the grid.
            Format: pixel = [i, j], where i = row and j = column

    """

    N, M = grid.shape
    max_energy = grid[0][0]
    pixel = [0,0]
    for i in range(N):
        for j in range(M):
            if max_energy > grid[i][j]:
                max_energy = max_energy
                pixel = pixel
            else:
                max_energy = grid[i][j]
                pixel = [i,j]
    return max_energy, pixel
# ------------------------------------------------------------------------------
def distance_formula(x, y, xhat, yhat):

    """
    Computes the distance between the points (x, y) and (xhat, yhat)

    INPUTS: (x, y) and (xhat, yhat)
    OUTPUT: ditance between these two points

    """
    import numpy as np
    return (np.sqrt( (yhat - y)**2 + (xhat - x)**2))
# ------------------------------------------------------------------------------
def find_starting_pixel(outputs):
    """
    This function finds the answers for the convolutional neural network.
    It finds the correct pixel that the electron started in. By taking the x and y
    coordinates of each electron and comparing it to a list of all the possible
    electron origins. When it finds a match it takes the number pixel that it
    started in. See file pixel_locations.txt for more information.

    INPUT: outputs = vector of all of the electron origins.
    OUTPUT: pixels = list of the pixel that the electron started in for its corresponding event

    """

    import sys
    sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python/electron_origins/src/')
    import setup_electron_densities as setup
    acceptable_ranges = setup.ranges()
    pixels = []
    for i in range(len(outputs)):
        for j in range(len(acceptable_ranges)):
            xmin = acceptable_ranges[j][0][0]
            xmax = acceptable_ranges[j][0][1]
            ymin = acceptable_ranges[j][1][0]
            ymax = acceptable_ranges[j][1][1]
            if xmin < outputs[i][1] < xmax and ymin < outputs[i][2] < ymax:
                pixels.append(j)
    return pixels
# ------------------------------------------------------------------------------
def print_stats(pixels, count_in, count_total):

    """
    This function is used to print the statistics from electron_densities.py.
    INPUTS: pixels = list of all of the starting pixels
            count_in = array of all of the electrons that fell inside its appropriate pixel
            count_total = array of all the times that this pixel was the pixel with the highest energy
    OUTPUTS: the statistics for each pixel.
                                 EXAMPLE
    Pixel                 Inside                Total                   P
    ----------------------------------------------------------------------
    [3,3]                  6000                 8000                   0.75
    and so on for all 100 of the starting pixels.

    """

    import numpy as np
    probabilites = []
    print("Pixel \t\t Inside \t Total \t\t P")
    print("-------------------------------------------------------------")
    for i in range(len(pixels)):
        p = count_in[i]/count_total[i]
        probabilites.append(p)
        print("%d,%d \t\t %d \t\t %d \t\t %.3f" %(pixels[i][0], pixels[i][1], count_in[i], count_total[i], p))
    print("--------------------------------------------------------------")
    print("Average: \t %d \t\t %d \t\t %.3f" %(np.floor(np.mean(count_in)), np.floor(np.mean(count_total)), np.mean(probabilites)))
