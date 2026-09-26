import python
import numpy
import math
import mediapipe as mp




def palme_center(pts):# pts is the array returned by mediapipe and contain the x/y cord of 21 hand tracking points
    pts = numpy.array(pts)
    ids = (0, 5, 9, 13, 17)#we will get the center by computing the average position of 5 specific landmarks
    cx=int(sum(pts[i][0] for i in ids)/len(ids))
    cy=int(sum(pts[i][1] for i in ids)/len(ids))
    return cx,cy
def hand_size(pts):
    pts = numpy.array(pts)
    radius = int(math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]))
    return radius*2
    #we calculate the radius by calculating the distance between the wrist and the base of the middle finger


def palm_orientation(pts):
    pts = numpy.array(pts)
    dx=pts[9][0] - pts[0][0]
    dy=pts[9][1] - pts[0][1]
    angle = math.degrees(math.atan2(dy,dx))+90
    return angle
def normalized_landmarks(pts):
    pts = numpy.array(pts)
    wrist_x,wrist_y=pts[0]
    scale = math.hypot(pts[9][0] - wrist_x, pts[9][1] - wrist_y)
    normalized=[((x-wrist_x)/scale,(y-wrist_y)/scale) for x,y in pts]
    return normalized



