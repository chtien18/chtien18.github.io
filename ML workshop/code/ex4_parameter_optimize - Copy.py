# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:27:30 2023

@author: FFL
"""

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.optimizers import RMSprop, Adam
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (
    plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

import tensorflow
import numpy as np
import pandas as pd
import dataset_preprocess
import logging


def my_func(x):

    row = x[:, 0].size
    fitness = np.zeros(row)

    for i in range(np.size(x[:, 0])):

        input_data = dataset_preprocess.normalize(x[i, :])
        # Load model
        #old_model_name = 'CV_1_470'
        old_model_name = 'CV_1_482'
        #epochs = 300
        #perf_avg = 100

        with open(f'{old_model_name}.json') as json_file:
            model_old = model_from_json(json_file.read())
            model_old.load_weights(f'{old_model_name}.h5')

        model = tensorflow.keras.models.clone_model(model_old)
        model.set_weights(model_old.get_weights())

        model.compile(
            optimizer=Adam(learning_rate=0.0005, beta_1=0.95, beta_2=0.999),
            loss='mae',
            metrics=['mse', 'mape']
        )

        y_hat = model.predict(input_data)
        y_hat_denorm = dataset_preprocess.denorm(y_hat)
        
        # ov_inlet = y_hat[0, 0]*189.249199
        # ov_outlet = y_hat[0, 1]*59.417836
        # ov_ratio = 1-ov_outlet/ov_inlet

        #fitness[i] = (y_hat[0,0] + y_hat[0,1] + ov_ratio)/3
        fitness[i] = (y_hat[0, 0] + y_hat[0, 1] +(1-y_hat_denorm[0,1]/y_hat_denorm[0,0]))/3
    return fitness

# main
# instatiate the optimizer


min_Ton = 3.5
min_Toff = 7
min_Ip = 0.5
min_WT = 0.2

max_Ton = 19.5
max_Toff = 39
max_Ip = 2.0
max_WT = 0.9

best_cost = 5
best_pos = []

x_min = [min_Ton, min_Toff, min_Ip, min_WT]
x_max = [max_Ton, max_Toff, max_Ip, max_WT]

bounds = (x_min, x_max)
options = {'c1': 5, 'c2': 5, 'w': 0.3}
optimizer = GlobalBestPSO(n_particles=20, dimensions=4,
                          options=options, bounds=bounds)

cost, pos = optimizer.optimize(my_func, 5000)

print('pos=', pos)

logging.info('best pos =' + str(pos))
cost_history = optimizer.cost_history

old_model_name = 'CV_1_482'
#epochs = 300
#perf_avg = 100

with open(f'{old_model_name}.json') as json_file:
    model_old = model_from_json(json_file.read())
    model_old.load_weights(f'{old_model_name}.h5')

model = tensorflow.keras.models.clone_model(model_old)
model.set_weights(model_old.get_weights())

model.compile(
    optimizer=Adam(learning_rate=0.0005, beta_1=0.95, beta_2=0.999),
    loss='mae',
    metrics=['mse', 'mape']
)
x = [pos[0], pos[1], pos[2], pos[3]]
input_data = dataset_preprocess.normalize(x)

y_hat = model.predict(input_data)
#y_hat_denorm = dataset_preprocess.denorm(y_hat)
print(y_hat)
print(np.average(y_hat))

df = pd.DataFrame(cost_history)
df.to_excel(excel_writer="CV1_482_results_history_30%_w1_w2_w3.xlsx")


