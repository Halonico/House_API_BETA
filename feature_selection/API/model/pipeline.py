# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import preprocessing as pp
from sklearn.impute import SimpleImputer
from category_encoders.one_hot import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
catToDiscreteDict = {
        "LotShape" : {"Reg" : 4,
               "IR1" : 3,
               "IR2" : 2,
               "IR3" : 1},
        "Utilities" : {  "AllPub" : 4,
               "NoSewr" : 3,
               "NoSeWa" : 2,
               "ELO" : 1,
               "None" : 0},
        "LandSlope" : {  "Gtl" : 3,
               "Mod" : 2,
               "Sev" : 1
            },
        "ExterQual" : {  "Ex" : 5,
               "Gd" : 4,
               "TA" : 3,
               "Fa" : 2,
               "Po" : 1,
            },
        "ExterCond" : {
                "Ex" : 5,
               "Gd" : 4,
               "TA" : 3,
               "Fa" : 2,
               "Po" : 1,
               "None" : 0
                },
        "BsmtQual" : {
               "Ex" : 5,
               "Gd" : 4,
               "TA" : 3,
               "Fa" : 2,
               "Po" : 1,
               "None" : 0
                },  
        "BsmtCond" : {
               "Ex" : 5,
               "Gd" : 4,
               "TA" : 3,
               "Fa" : 2,
               "Po" : 1,
               "None" : 0
                },
        "BsmtExposure" :  {
               "Gd" : 4,
               "Av" : 3,
               "Mn" : 2,
               "No" : 1,
               "None" : 0
            },
        "BsmtFinType1" : {
                
               "GLQ" : 6,
               "ALQ" : 5,
               "BLQ" : 4,
               "Rec" : 3,
               "LwQ" : 2,
               "Unf" : 1,
               "None": 0
            },
        "BsmtFinType2" : {
                "GLQ" : 6,
               "ALQ" : 5,
               "BLQ" : 4,
               "Rec" : 3,
               "LwQ" : 2,
               "Unf" : 1,
               "None": 0
                },
        "HeatingQC" : {
                "Ex" : 5,
                "Gd" : 4,
                "TA" : 3,
                "Fa" : 2,
                "Po" : 1,
                },
        "KitchenQual" : {
                "Ex" : 5,
                "Gd" : 4,
                "TA" : 3,
                "Fa" : 2,
                "Po" : 1,
                "None" : 0
                },
        "Functional" : {
                "Typ" : 8,
                "Min1" : 7,
                "Min2" : 6,
                "Mod" : 5,
                "Maj1" : 4,
                "Maj2" : 3,
                "Sev" : 2,
                "Sal" : 1,
                "None" : 0
                },
        "GarageFinish" : {
                "Fin" : 3,
                "RFn" : 2,
                "Unf" : 1,
                "None" : 0
                },
        "GarageQual" : {
                 "Ex" : 5,
                 "Gd" : 4,
                 "TA" : 3,
                 "Fa" : 2,
                 "Po" : 1,
                 "None" : 0
                },
        "GarageCond" : {
                 "Ex" : 5,
                 "Gd" : 4,
                 "TA" : 3,
                 "Fa" : 2,
                 "Po" : 1,
                 "None" : 0
                },
        "PavedDrive" : {
                "Y" : 3,
                "P" : 2,
                "N" : 1
                },
        "PoolQC" : {
                "Ex" : 4,
                "Gd" : 3,
                "TA" : 2,
                "Fa" : 1,
                "None" : 0
        },
        "FireplaceQu" : {
                "Ex" : 5,
                "Gd" : 4,
                "TA" : 3,
                "Fa" : 2,
                "Po" : 1,
                "None" : 0
        },
        "Fence" : {
        "GdPrv" : 4,
        "MnPrv" : 3,
        "GdWo" : 2,
        "MnWw" : 1,
        "None" : 0
        },
        
        }
realCategories = {"MSSubClass" : [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190],
                   "MSZoning" : [ "A","C","FV","I","RH","RL","RP","RM"],
                   "Neighborhood" : ["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","Names","NoRidge","NPkVill","NridgHt"],
                   "HouseStyle" : ["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"],
                   "Exterior1st" : ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShing"],
                   "Exterior2nd" : ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard","ImStucc","MetalSd","Other","Plywood","PreCast","Stone","Stucco","VinylSd","Wd Sdng","WdShing"],
                   "MasVnrType" : ["BrkCmn","BrkFace","CBlock","None","Stone"],
                   "Foundation" : ["BrkTil","CBlock","PConc","Slab","Stone","Wood"],
                   "GarageType" : ["2Types","Attchd","Basment","BuiltIn","CarPort","Detchd","None"]
                   }
dateCols = ["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"]
categories1and0 = {
        "Street" : "Pave",
        "Alley" : "None",
        "LotConfig" : "Inside",
        "Condition1" : "Norm",
        "Condition2" : "Norm",
        "BldgType" : "1Fam",
        "RoofStyle" : "Gable",
        "RoofMatl" : "CompShg",
        "CentralAir" : "Y",
        "Heating" : "GasA",
        "Electrical" : "SBrkr",
        "MiscFeature" : "None",
        "SaleType" : "WD",
        "SaleCondition" : "Normal",
        "LandContour" : "Lvl"
        }
featuresFill0 = ["LotFrontage","MasVnrArea","BsmtHalfBath","BsmtFullBath","BsmtFinSF1","BsmtFinSF2","GarageArea","TotalBsmtSF","BsmtUnfSF","GarageCars"]
featureFillMedian= ["GarageYrBlt"]
prediction_pipeline = Pipeline([
    ('fillNaNCategorical',pp.FillNaNCategoricalWithNone()),
    ('transformCategoryToDiscrete',pp.CategoricalToDiscrete(catToDiscreteDict)),
    ('transformCategoryTo1and0',pp.CategoricalTo0and1(categories1and0)),
    ('ReplaceValueBy0',pp.ReplaceValueBy0(featuresFill0)),
    ('ReplaceValueByMedian',pp.ReplaceValueByMedian(featureFillMedian)),
    ('ConvertDummyFeatures',pp.CategoryToDummies(realCategories)),
    ('ConvertYearFeatureToAbsolute', pp.YearToAbsolute(dateCols)),
    ('StandardizeFeatures', pp.StandardizeFeatures()),
    #('PCAReduction', pp.PCAReduction()),
    #('ModelPrediction',XGBRegressor())
    ('ModelPrediction',RandomForestRegressor(max_depth=6,n_estimators=100))
])