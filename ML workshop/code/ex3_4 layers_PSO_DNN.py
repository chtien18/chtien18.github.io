# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:36:39 2023

@author: chtie
"""
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.optimizers import RMSprop, Adam
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import dataset_preprocess


def dnn_training(x):
    D1 = x[:,0]
    DO1 = x[:,1]
    D2 = x[:,2]
    DO2 = x[:,3]
    D3 = x[:,4]
    DO3 = x[:,5]
    D4 = x[:,6]
    DO4 = x[:,7]
    RSN = x[:,8]
    
    row = D1.size
    fitness = np.zeros(row)
    scores_mse_train = []
    scores_mape_train =[]
    scores_mse_test = []
    scores_mape_test =[]
    scores_mse_val = []
    scores_mape_val =[]
    scores_mape =[]
    
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
        )
    for i in range(row):
        Seed = round(RSN[i])
        D1_ = round(D1[i])
        DO1_= round(DO1[i],2)
        D2_ = round(D2[i])
        DO2_= round(DO2[i],2)
        D3_ = round(D3[i])
        DO3_= round(DO3[i],2)
        D4_ = round(D4[i])
        DO4_= round(DO4[i],2)
        global best_cost
        global best_pos
        print('best_cost=',best_cost)
        print('best_pos=',best_pos)
        #data normalization from excel
        data = pd.read_excel('result_normalize.xlsx')
        #data_shuffled = data.sample(frac=1, random_state=(Seed))

        X = data[['Ton_norm','Toff_norm','LV_norm','WT_norm']]
        Y = data[['Inlet_Diameter_norm','Outlet_Diameter_norm']]
        X=X.to_numpy()
        Y=Y.to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/6, shuffle=(False))
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=(Seed))

        epochs = 400

        model = keras.Sequential([
            layers.Dense(D1_, activation='relu',input_shape=[4]),
            layers.Dropout(rate=DO1_),
            layers.Dense(D2_, activation='relu'),
            layers.Dropout(rate=DO2_),
            layers.Dense(D3_, activation='relu'),
            layers.Dropout(rate=DO3_),
            layers.Dense(D4_, activation='relu'),
            layers.Dropout(rate=DO4_),
            layers.Dense(units=2,activation='relu')
            ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005,beta_1=0.95,beta_2=0.999),
            #optimizer=RMSprop(learning_rate=0.001),
            
            loss='mae',
            metrics=['mse','mape']
            )

       # Fit the model

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val), 
            epochs=epochs, 
            callbacks=[early_stopping],
            verbose=0)

        history_df = pd.DataFrame(history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot();
        print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
        plt.show()

        scores_train = model.evaluate(x_train, y_train, verbose=0)
        scores_val = model.evaluate(x_val, y_val, verbose=0)
        scores_test = model.evaluate(x_test, y_test, verbose=0)

        print((model.metrics_names, scores_train))
        print((model.metrics_names, scores_val))
        print((model.metrics_names, scores_test))

        scores_mse_train.append(scores_train[1])
        scores_mse_val.append(scores_val[1])
        scores_mse_test.append(scores_test[1])


        # scores_mape_train=scores_train[2]
        # scores_mape_val=scores_val[2]
        # scores_mape_test=scores_test[2]

        #fitness = (scores_mape_train + scores_mape_val + scores_mape_test )/3
        # print('fitness=',fitness)

        y_train_hat = model.predict(x_train)
        y_val_hat = model.predict(x_val)
        y_test_hat = model.predict(x_test)

        y_train_hat_denorm = dataset_preprocess.denorm(y_train_hat)
        y_val_hat_denorm = dataset_preprocess.denorm(y_val_hat)
        y_test_hat_denorm = dataset_preprocess.denorm(y_test_hat)

        y_train_denorm = dataset_preprocess.denorm(y_train)
        y_val_denorm = dataset_preprocess.denorm(y_val)
        y_test_denorm = dataset_preprocess.denorm(y_test)

        scores_mape_train = 100*mean_absolute_percentage_error(y_train_denorm, y_train_hat_denorm)
        scores_mape_val = 100*mean_absolute_percentage_error(y_val_denorm, y_val_hat_denorm)
        scores_mape_test = 100*mean_absolute_percentage_error(y_test_denorm, y_test_hat_denorm)
        
        print('scores_mape_train=',scores_mape_train)
        print('scores_mape_val=',scores_mape_val)
        print('scores_mape_test=',scores_mape_test)
        fitness[i] = (scores_mape_train + scores_mape_test + scores_mape_val)/3
        print('fitness=',fitness[i])
        
        if fitness[i]<best_cost:
            best_cost = fitness[i]
            best_pos = [D1_, DO1_, D2_, DO2_, D3_, DO3_, D4_, DO4_, Seed]
            model_json = model.to_json()
            with open("pre_train_noCV.json", "w") as json_file:
                json_file.write(model_json)
                model.save_weights("pre_train_noCV.h5")
        
    return fitness



#main
# instatiate the optimizer

min_D = 0
min_DO = 0 
min_RSN = 1

max_D = 200 
max_DO = 0.5 
max_RSN = 200 
best_cost =5
best_pos =[]

x_min = [min_D, min_DO, min_D, min_DO, min_D, min_DO, min_D, min_DO, min_RSN]
x_max = [max_D, max_DO, max_D, max_DO, max_D, max_DO, max_D, max_DO, max_RSN]
bounds = (x_min, x_max)
options = {'c1': 5, 'c2': 5, 'w': 0.5}
optimizer = GlobalBestPSO(n_particles=10, dimensions=9, options=options, bounds=bounds)

cost, pos = optimizer.optimize(dnn_training, 500)


print('pos=',pos)

cost_history = optimizer.cost_history
