import numpy as np


def LocalMaxima(res, row, col):
    '''
    1,2,3
    4, ,5
    6,7,8
    '''
    maxDim = res.shape
    # 1
    if row >= 1 and col >= 1:
        if res[row, col] < res[row-1, col-1]:
            return False
    # 2
    if row >= 1:
        if res[row, col] < res[row-1, col]:
            return False
    # 3
    if row >= 1 and col < maxDim[1]-1:
        if res[row, col] < res[row-1, col+1]:
            return False

    # 4
    if col >= 1:
        if res[row, col] < res[row, col-1]:
            return False

    # 5
    if col < maxDim[1]-1:
        if res[row, col] < res[row, col+1]:
            return False

    # 6
    if row < maxDim[0] - 1 and col >= 1:
        if res[row, col] < res[row+1, col-1]:
            return False

    # 7
    if row < maxDim[0] - 1:
        if res[row, col] < res[row+1, col]:
            return False

    # 8
    if row < maxDim[0] - 1 and col < maxDim[1]-1:
        if res[row, col] < res[row+1, col+1]:
            return True

    return True


def CalcAccuracyPoint(resMap, row, col):
    resSize = resMap.shape
    if row == 0 or col == 0 or row == resSize[0]-1 or col == resSize[1]-1:
        return row, col, 0

    # 1. calc dx dy
    dx = 0.5*(resMap[row, col+1]-resMap[row, col-1])
    dy = 0.5*(resMap[row+1, col]-resMap[row-1, col])
    # 2. calc dxx dyy dxy
    dxx = resMap[row, col+1]+resMap[row, col-1]+2*resMap[row, col]
    dyy = resMap[row+1, col]+resMap[row-1, col]+2*resMap[row, col]
    dxy = 0.25*(resMap[row+1, col+1]+resMap[row-1, col-1]
                - resMap[row+1, col-1] - resMap[row-1, col+1])
    # 3. combine full array
    deriv2 = np.array(([dxx, dxy], [dxy, dyy]))
    deriv = np.array([dx, dy])
    # 4. calc inv(ddx)
    invderiv2 = np.linalg.inv(deriv2)
    # 5. offset
    offset = -1*np.matmul(invderiv2, deriv)

    return offset


def IntersectionArea(r11, c11, r12, c12, r21, c21, r22, c22):
    x_overlap = Math.max(0, Math.min(
        c12, c22) - Math.max(c11, c21))
    y_overlap = Math.max(0, Math.min(
        r12, r22) - Math.max(r11, r21))
    overlapArea = x_overlap * y_overlap
    return overlapArea
