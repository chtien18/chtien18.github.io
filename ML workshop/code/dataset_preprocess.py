
import numpy as np


import pandas as pd
import numpy as np




import os

def normalize(X):
    min_ton = 4#5
    min_toff = 8 #10
    min_lv =0.5#0.5
    min_wt = 0.2 #0.3
    
    max_ton = 18 #15
    max_toff = 36 #30
    max_lv = 2 #1.5
    max_wt = 0.9 #0.7
    
    Ton = (X[0] - min_ton)/(max_ton - min_ton)
    Toff = (X[1] - min_toff)/(max_toff - min_toff)
    LV = (X[2] - min_lv)/(max_lv - min_lv)
    WT = (X[3] - min_wt)/(max_wt - min_wt)
 
    data1={'Ton':pd.Series(Ton),'Toff':pd.Series(Toff),
          'LV':pd.Series(LV),
          'WT':pd.Series(WT)}
    
    x_norm = pd.DataFrame(data1)
    x_norm=x_norm.to_numpy()
    
    return x_norm

def dataset_prep(RSN):
    min_ton = 4#5
    min_toff = 8 #10
    min_lv =0.5#0.5
    min_wt = 0.2 #0.3
    
    max_ton = 18 #15
    max_toff = 36 #30
    max_lv = 2 #1.5
    max_wt = 0.9 #0.7
    
    #define dataset
    data = pd.read_excel('result_normalize.xlsx')
    
    data_shuffled = data.sample(frac=1, random_state=(RSN))
    
    X = data_shuffled[['Ton','Toff','LV','WT']]
    Y= data_shuffled[['Inlet_Diameter','Outlet_Diameter']]
    
    #X = data[['Ton','Toff','LV','WT']]
    #Normalize data, max min from the maching setup
    Ton = (X['Ton'] - min_ton)/(max_ton - min_ton)
    Toff = (X['Toff'] - min_toff)/(max_toff - min_toff)
    LV = (X['LV'] - min_lv)/(max_lv - min_lv)
    WT = (X['WT'] - min_wt)/(max_wt - min_wt)
    
 
    data1={'Ton':pd.Series(Ton),'Toff':pd.Series(Toff),
          'LV':pd.Series(LV),
          'WT':pd.Series(WT)}
    
    x_norm = pd.DataFrame(data1)
    
    #Y= data[['Inlet_Diameter','Outlet_Diameter']]
    
    min_Inlet_Diameter=300
    min_Outlet_Diameter=300
    
    #max_Inlet_Diameter = 489.25
    #max_Outlet_Diameter= 376.84
    max_Inlet_Diameter=max(Y['Inlet_Diameter'])
    #print(max_Inlet_Diameter)
    max_Outlet_Diameter=max(Y['Outlet_Diameter'])
    #print(max_Outlet_Diameter)
    # max_Inlet_roundness=max(Y['Inlet_roundness'])
    # max_Outlet_roundness=max(Y['Outlet_roundness'])
    
    #Normalize data, max min from the maching setup
    Inlet_Diameter = (Y['Inlet_Diameter'] - min_Inlet_Diameter)/(max_Inlet_Diameter - min_Inlet_Diameter)
    Outlet_Diameter = (Y['Outlet_Diameter'] - min_Outlet_Diameter)/(max_Outlet_Diameter - min_Outlet_Diameter)
    # Inlet_roundness = (Y['Inlet_roundness'] - min_Inlet_roundness)/(max_Inlet_roundness - min_Inlet_roundness)
    # Outlet_roundness = (Y['Outlet_roundness'] - min_Outlet_roundness)/(max_Outlet_roundness - min_Outlet_roundness)
    
    
    # data2={'Inlet_Diameter':pd.Series(Inlet_Diameter),'Outlet_Diameter':pd.Series(Outlet_Diameter),
    #        'Inlet_roundness':pd.Series(Inlet_roundness),'Outlet_roundness':pd.Series(Outlet_roundness)}
    
    data2={'Inlet_Diameter':pd.Series(Inlet_Diameter),'Outlet_Diameter':pd.Series(Outlet_Diameter)}
    
    
    y_norm=pd.DataFrame(data2)

    x_norm=x_norm.to_numpy()
    y_norm=y_norm.to_numpy()
    
    return x_norm, y_norm

def denorm(y):
    #define dataset
    data = pd.read_excel('result_normalize.xlsx')
    
    data_shuffled = data.sample(frac=1, random_state=0)

    Y= data_shuffled[['Inlet_Diameter','Outlet_Diameter']]
    
    
    min_Inlet_Diameter=300
    min_Outlet_Diameter=300
    
    max_Inlet_Diameter=max(Y['Inlet_Diameter'])
    #print(max_Inlet_Diameter)
    max_Outlet_Diameter=max(Y['Outlet_Diameter'])
    #print(max_Outlet_Diameter)
    
    #Normalize data, max min from the maching setup
    Inlet_Diameter = min_Inlet_Diameter + y[:,0]*(max_Inlet_Diameter - min_Inlet_Diameter)
    Outlet_Diameter = min_Outlet_Diameter + y[:,1]*(max_Outlet_Diameter - min_Outlet_Diameter)

    data3={'Inlet_Diameter':pd.Series(Inlet_Diameter),'Outlet_Diameter':pd.Series(Outlet_Diameter)}
    
    y_denorm=pd.DataFrame(data3)

    y_denorm=y_denorm.to_numpy()
    
    return y_denorm
