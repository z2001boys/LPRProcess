

def MaxIdx(v):
    maxv = 0
    maxi = 0
    for i in range(len(v)):
        if maxv < v[i]:
            maxv = v[i]
            maxi = i
    return maxv,maxi