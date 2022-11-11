---
layout: page-fullwidth
title:  " Machine learning model"
subheadline:  " "
teaser: " Machine learning model"
categories: 
   - study
header:
   image_fullwidth: header_unsplash_leaf.jpg

image: header_unsplash_leaf.jpg
---

### Machine learning model
Buid a basic machine learning model using scikit-learn library.
Loading iowa_housing_data (download).

    # Path of the file to read
    iowa_file_path = '../train.csv'

    home_data = pd.read_csv(iowa_file_path)
    home_data.columns
Result:

    Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
    'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
    'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
    'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
    'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
    'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
    'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
    'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
    'SaleCondition', 'SalePrice'],
    dtype='object')
The dataset have too many variables. We will pick a few variables using our intuition to build a prediction model.

Select the target variable, which is the sales price. Save this to a new variable called y.

    y = home_data['SalePrice']
Create input feature - X
Creating a DataFrame called X holding the predictive features.
We will use the following features instead of the whole data features to make a prediction:

* LotArea
* YearBuilt
* 1stFlrSF
* 2ndFlrSF
* FullBath
* BedroomAbvGr
* TotRmsAbvGrd
    #Create the list of features
    feature_names = {'LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd'}

Select data corresponding to features in feature_names

    X = home_data[feature_names]

Create and train a model
We will create a DecisionTreeRegressor model.

    from sklearn.tree import DecisionTreeRegressor
    #specify the model. 
    #For model reproducibility, set a numeric value for random_state when specifying the model
    iowa_model = DecisionTreeRegressor(random_state =1)

    # Fit the model
    iowa_model.fit(X,y) 

Make a prediction
After training, we can make a prediction from trained model using predict command. In this case, we make a prediction using X as input data.

    predictions = iowa_model.predict(X)
    print(predictions) # the prediction sale price
    print(y.head()) # the actual sale price

Result

    [208500. 181500. 223500. ... 266500. 142125. 147500.]
    0    208500
    1    181500
    2    223500
    3    140000
    4    250000
    Name: SalePrice, dtype: int64
As shown in the result, we can compare the prediction and actual sale price that they are exactly the same. It is because the data were used in the training process.

### Model validation
We will test how good the model is. It means that how good the model be able to predict unknown data.
For this part, the iowa_house_data is splited into two sub-datasets which are training set and validation set.

First, create input X and output y

    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor

    # Path of the file to read
    iowa_file_path = '../train.csv'

    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[feature_columns]
Split data into two set: training_set and validation_set

    # Import the train_test_split function
    from sklearn.model_selection import train_test_split

    # train the model
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)
The training set is (train_X, train_y) used for training process. The validation set is (val_X, val_y), which is unknown-data, used for validation process.

Train model

    # Specify the model
    iowa_model = DecisionTreeRegressor(random_state=1)

    # Fit iowa_model with the training data.
    iowa_model.fit(train_X, train_y)
    Make prediction with validation dataset

    # Predict with all validation observations
    val_predictions = iowa_model.predict(val_X)
    Print out the prediction and actual values from validation data

    # print the top few validation predictions
    print(val_predictions[0:5])
    # print the top few actual prices from validation data
    print(val_y[0:5].tolist())
result:

    [186500. 184000. 130000.  92000. 164500.]
    [231500, 179500, 122000, 84500, 142000]
We calculate Mean_absolute_error between prediction and actual values to evaluate the model performance.

    from sklearn.metrics import mean_absolute_error
    val_mae = mean_absolute_error(val_y, val_predictions)
    print(val_mae)
result:

    29652.931506849316 
It means that the average diffrence between prediction and the actual house price is about 30,000 dollars.
There are many ways to improve the model, such as experimenting to find better features or different model types.