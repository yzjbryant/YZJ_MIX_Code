import matplotlib.pyplot as plt
import numpy as np
e = [50.5, 32.4466, 139, 63.18]
s = [3.695, 2.7205, 2.42, 3.5]
qi = [0, 0, -0.115, 0.115]
qj = [0, 1, -0.115, 0.115]
e0 = 8.854187817e-12
x = np.linspace(0.000001, 25.6, 100).tolist()
for i in range(len(e)):
    y = [4 * e[i] * ((s[i] / t)**12 - (s[i] / t)**6) + qi[i] * qj[i] / 4 / np.pi / e0 / t for t in x]
    plt.plot(x, y, 'b')
    plt.show()