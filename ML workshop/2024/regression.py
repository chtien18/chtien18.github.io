import pandas as pd
import numpy as np


from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.optimizers import RMSprop, Adam

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import dataset_preprocess

# early_stopping = callbacks.EarlyStopping(
#     min_delta=0.001, # minimium amount of change to count as an improvement
#     patience=50, # how many epochs to wait before stopping
#     restore_best_weights=True,
# )

Seed = 100
D1_ = 60
DO1_= 0.0
D2_ = 46
DO2_= 0.09
D3_ = 131
DO3_= 0.33
D4_ = 49
DO4_= 0.36
D5_ = 100
DO5_= 0.2
D6_ = 100
DO6_= 0.2

print('D1=',D1_)
print('DO1=',DO1_)
print('D2=',D2_)
print('DO2=',DO2_)
print('D3=',D3_)
print('DO3=',DO3_)
print('RSN=',Seed)


scores_mse_train = []
scores_mape_train =[]
scores_mse_test = []
scores_mape_test =[]
scores_mse_val = []
scores_mape_val =[]
scores_mape =[]

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
    
    loss='mse',
    metrics=['mse','mape']
    )

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val), 
    epochs=epochs, 
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

fitness = (scores_mape_train + scores_mape_val + scores_mape_test )/3
print('fitness=',fitness)

plt.plot(y_train_denorm[:,0],y_train_denorm[:,0])
plt.scatter(y_train_denorm[:,0],y_train_hat_denorm[:,0])
plt.show()

plt.plot(y_train_denorm[:,0])
plt.plot(y_train_hat_denorm[:,0])
plt.show()

plt.plot(y_train_denorm[:,1])
plt.plot(y_train_hat_denorm[:,1])
plt.show()

