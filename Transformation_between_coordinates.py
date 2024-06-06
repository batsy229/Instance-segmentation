import numpy as np
import math

cosAngle = math.cos(math.radians(-145))
sinAngle = math.sin(math.radians(-145))
rotMatrix = np.array([[1, 0, 0],
                      [0, cosAngle, -sinAngle],
                      [0, sinAngle, cosAngle]])
#55,15,382
#65, 26, 442
point = np.array([[250], [5], [400]])
translation = np.array([[-165], [188], [-73]])
# translation = np.array([[-160], [160], [-73]])

yT = (point[1]*cosAngle) - (point[2]*sinAngle)
zT = (point[1]*sinAngle) + (point[2]*cosAngle)

pointT = np.array([point[0], yT, zT])

# pointT = np.matmul(rotMatrix, point)

pointF = np.add(pointT, translation)

print(pointT)
print(pointF)