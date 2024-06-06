from random import randint
import time


class MyFace:
    tracks = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.name = []
        self.staff_id = []
        self.confidence = []
        self.age = 0
        self.max_age = max_age
        self.show = False

    def getRGB(self):
        return (self.R, self.G, self.B)

    def getTracks(self):
        return self.tracks

    def getShow(self):
        return self.show

    def getId(self):
        return self.i

    def getDone(self):
        return self.done

    def getX(self):
        return self.x

    def getAge(self):
        return self.age

    def getName(self):
        return self.name

    def getStaffID(self):
        return self.staff_id

    def getY(self):
        return self.y

    def setDone(self):
        self.done = True

    def setShow(self):
        if self.age > 5:
            self.show = True

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True

    def getConfidence(self):
        return self.confidence

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True

    def updateCoords(self, xn, yn):
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def updateStaffID(self, staff_id):
        self.staff_id.append(staff_id)

    def updateName(self, name):
        self.name.append(name)

    def updateConfidence(self, confidence):
        self.confidence.append(confidence)
