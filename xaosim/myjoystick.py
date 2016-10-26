import math
import pygame
import sys

# ============== was defined in the helpers class ==============

def limitToRange(a,b,c):
    if a < b:
        a = b
    if a > c:
        a = c
    return a

# ==============================================================

class Joystick:
    def __init__(self):
        pygame.joystick.init()
        numJoys = pygame.joystick.get_count()

        if (numJoys == 1):
            joyI = 0
        else:
            sys.stdout.write("No joystick connected\n")
            return(None)

        self.joystick = pygame.joystick.Joystick(joyI)
        self.joystick.init()
        self.numButtons = self.joystick.get_numbuttons()

        if self.numButtons == 0:
            sys.stdout.write("No joystick connected\n")
            return(None)
                    
        self.buttons    = [0]*self.numButtons
        self.naxes      = self.joystick.get_numaxes()
        self.nhats      = self.joystick.get_numhats()
        self.hats       = [0] * self.nhats;

        self.x1 = 0
        self.y1 = 0
        self.rad1 = 0

        self.x2 = 0
        self.y2 = 0
        self.rad2 = 0

    def compute(self):
        self.x1 = self.joystick.get_axis(0)
        self.y1 = self.joystick.get_axis(1)
        self.rad1 = math.hypot(self.x1,self.y1)
        self.rad1 = limitToRange(self.rad1,0,1)
        self.ang1 = math.atan2(self.y1,self.x1)
        self.x1 = self.rad1*math.cos(self.ang1)
        self.y1 = self.rad1*math.sin(self.ang1)

        self.x2 = self.joystick.get_axis(2)
        self.y2 = self.joystick.get_axis(3)
        self.rad2 = math.hypot(self.x2, self.y2)
        self.rad2 = limitToRange(self.rad2, 0, 1)
        self.ang2 = math.atan2(self.y2,self.x2)
        self.x2 = self.rad2*math.cos(self.ang2)
        self.y2 = self.rad2*math.sin(self.ang2)

        #'clicks' to middle
        tab = .12
        if -tab < self.x1 < tab:
            self.x1 = 0
        if -tab < self.y1 < tab:
            self.y1 = 0
