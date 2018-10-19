
import math

def calPointByAng( r,c,ang,len ):
    row = -1*math.sin(ang)*len+r
    col = math.cos(ang)*len+c
    return row,col


def GetArrow( r,c,ty,amp ):
    ele = math.pi/8
    angle = [
        ele*0,ele*1,ele*2,ele*3,ele*4,ele*5,ele*6,ele*7
    ]
    row,col = calPointByAng(r,c,angle[ty],amp)
    return row,col