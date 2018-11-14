import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# a = np.array([[1,    8,   50],
#  [8,   64,  400],
#  [50,  400, 2500]])
# print(np.linalg.det(a))
# print(inv(a))

X = np.arange(12)
plt.plot(X, [-27.80190177509302, -18.278008698672398, -14.155975751698152, -9.355509112700402, -6.928485221001136,
             -7.252783769260551, -9.075566102641499, -12.217659850459722, -15.762571764008012, -19.10983337936544,
             -21.26188672609279,  -19.165376585714384])
plt.ylabel('maximum of the log marginal likelihood')
plt.xlabel('orders')
plt.show()
print("done")

#
# The local maximum for order  0 is  -27.80190177509302
#
# The local maximum for order  1 is  -18.278008698672398
#
#
# The local maximum for order  1 is  -18.278008698672398
#
#
# The local maximum occurs at [0.1713283  0.09402557]
# The local maximum for order  2 is  -14.155975751698152
#
#
# The local maximum occurs at [0.13241786 0.0432884 ]
# The local maximum for order  3 is  -9.355509112700402
#
#
# The local maximum occurs at [0.10693263 0.0230254 ]
# The local maximum for order  4 is  -6.928485221001136
#
# The local maximum for order  5 is  -7.252783769260551 X
#
# The local maximum for order  6 is  -9.075566102641499 X
#
# The local maximum for order  7 -12.217659850459722
#
# The local maximum for order  8 is  -15.762571764008012
#
# The local maximum occurs at [0.05257424 0.01684657]
# The local maximum for order  9 is  -19.10983337936544
#
#
#
# The local maximum occurs at [0.05077136 0.01131081]
# The local maximum for order  10 is  -21.26188672609279
#
#
# The local maximum occurs at [0.04474178 0.00024963]
# The local maximum for order  11 is  -19.165376585714384
