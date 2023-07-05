# Import modules
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.single.global_best import GlobalBestPSO


def my_function1(x):
    x1 = x[:,0]
    x2 = x[:,1]
    y = 4*x1*x1 -2.1*x1**4 +1/3*x1**6 +x1*x2 - 4*x2*x2+4*x2**4
    return y

# instatiate the optimizer
x_max = 10 * np.ones(2)
x_min = -1 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)

cost, pos = optimizer.optimize(my_function1, 200)


plot_cost_history(cost_history=optimizer.cost_history)
plt.show()