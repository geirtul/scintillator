# CATEGORIZE A HARD TWO ELECTRON CASE AND AN EASY ONE
# ------------------------------------------------------------------------------
# This code sorts through easy and hard cases in the two electron data. It does
# this by counting how many blocks of lit up pixels there are. If there are more
# than one block then this is an easy case, because it is clear where the
# two electrons were. If there is single cluster of pixels, then this is a hard
# case, because it might be hard for the machine to recognize this as a two electron
# event. This also provides us with a performance measure of how well our neural
# network should perform
# ------------------------------------------------------------------------------
# Authors: Harrison, Yani, Alex

import numpy as np
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/python')
import ProcessingData as data
filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/MSU REU/MSU REU Project/BetaScint2DEnergy2Electron.csv')

grid, outputs = data.get_data(filename)


def excluded(M, i,j):
    if M[i][j] == 1:
        return True

def neighbors(E, M, i, j, count):
    count = count
    M = M
    if i-1 >=0 and i-1 < N and j-1 >=0 and j-1 < N: #upper left diagonal

        if M[i-1][j-1] == 0:
            if E[i-1][j-1] > 0:
                M[i-1][j-1] = 1
                count +=1
                M, count = neighbors(E,M,i-1, j-1,count)
            else:
                M[i-1][j-1] = 2
    if i-1 >=0 and i-1 < N and j >=0 and j < N: #up neighbor

        if M[i-1][j] == 0:
            if E[i-1][j] > 0:
                M[i-1][j] = 1
                count +=1
                M, count = neighbors(E,M,i-1, j,count)
            else:
                M[i-1][j] = 2
    if i-1 >=0 and i-1 < N and j+1 >=0 and j+1 < N: #upper right diagonal

        if M[i-1][j+1] == 0:
            if E[i-1][j+1] > 0:
                M[i-1][j+1] = 1
                count +=1
                M, count = neighbors(E,M,i-1, j+1,count)
            else:
                M[i-1][j+1] = 2
    if i >=0 and i < N and j-1 >=0 and j-1 < N: #left neighbor

        if M[i][j-1] == 0:
            if E[i][j-1] > 0:
                M[i][j-1] = 1
                count +=1
                M, count = neighbors(E,M,i, j-1,count)
            else:
                M[i][j-1] = 2
    if  i >= 0 and i < N and j+1 >=  0 and j+1 < N: #right neighbor

        if M[i][j+1] == 0:
            if E[i][j+1] > 0:
                M[i][j+1] = 1
                count +=1
                M, count = neighbors(E,M,i, j+1,count)
            else:
                M[i][j+1] = 2
    if  i+1 >= 0 and i+1 < N and j-1 >=  0 and j-1 < N: #lower left diagonal

        if M[i+1][j-1] == 0:
            if E[i+1][j-1] > 0:
                M[i+1][j-1] = 1
                count +=1
                M, count = neighbors(E,M,i+1, j-1,count)
            else:
                M[i+1][j-1] = 2
    if  i+1 >= 0 and i+1 < N: #down neighbor
        if M[i+1][j] == 0:
            if E[i+1][j] > 0:
                M[i+1][j] = 1
                count +=1
                M, count = neighbors(E,M,i+1, j,count)
            else:
                M[i+1][j] = 2
    if  i+1 >= 0 and i+1 < N and j+1 >=  0 and j+1 < N: #lower right diagonal

        if M[i+1][j+1] == 0:
            if E[i+1][j+1] > 0:
                M[i+1][j+1] = 1
                count +=1
                M, count = neighbors(E,M,i+1, j+1,count)
            else:
                M[i+1][j+1] = 2
    return M, count


hard = [] # hard cases will be when len(v) != 2
for k in range(len(grid)):
    test = grid[k]
    N, M = test.shape
    checker = np.zeros((N,M))
    v = []
    for i in range(N):
        for j in range(M):

            if excluded(checker, i, j) == True:
                continue
            count = 0
            if test[i][j] > 0:
                checker, count = neighbors(test, checker, i, j, count)
                if count == 0:
                    count+= 1
                v.append(count)
    if len(v) != 2:
        hard.append(grid[k])

print(len(hard)/len(grid))

# We found that 15 % of the cases are hard. So the CNN should perform very well.
