# House_API_BETA

## Description

This project is a basic api with no database linked to it.
It allows the users to train a model based on this competition : https://www.kaggle.com/c/house-prices-advanced-regression-techniques
And make a prediction from a json object

## Routes

### GET http://127.0.0.1:5000/state

Return a string that says if the server is running
### GET http://127.0.0.1:5000/state

Return a string that says "hello"

### POST http://127.0.0.1:5000/prediction

Make a prediction from a json object in the body.
Example of payload : 

```
    [
  {
    "MSSubClass": 20,
    "MSZoning": "RH",
    "LotFrontage": 80,
    "LotArea": 11622,
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Feedr",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "OverallQual": 5,
    "OverallCond": 6,
    "YearBuilt": 1961,
    "YearRemodAdd": 1961,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "CBlock",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Rec",
    "BsmtFinSF1": 468,
    "BsmtFinType2": "LwQ",
    "BsmtFinSF2": 144,
    "BsmtUnfSF": 270,
    "TotalBsmtSF": 882,
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 896,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 896,
    "BsmtFullBath": 0,
    "BsmtHalfBath": 0,
    "FullBath": 1,
    "HalfBath": 0,
    "BedroomAbvGr": 2,
    "KitchenAbvGr": 1,
    "KitchenQual": "TA",
    "TotRmsAbvGrd": 5,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": "None",
    "GarageType": "Attchd",
    "GarageYrBlt": 1961,
    "GarageFinish": "Unf",
    "GarageCars": 1,
    "GarageArea": 730,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 140,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 120,
    "PoolArea": 0,
    "PoolQC": "None",
    "Fence": "MnPrv",
    "MiscFeature": "None",
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2010,
    "SaleType": "WD",
    "SaleCondition": "Normal"
  }]


```

## How to run it
1) Git clone the repository
2) Go in the feature_selection's folder
3) Run in cmd : ```pip install -r requirements.txt```
4) Run in cmd :  ```python run.py```

There is also a python virtualenvironment included
