import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def load_data():
    train = pd.read_csv("D:/Projects/Ya/intelligent-learning/machine-learning/tasks/house-price/dataset/house-prices-advanced-regression-techniques/train.csv")
    test = pd.read_csv("D:/Projects/Ya/intelligent-learning/machine-learning/tasks/house-price/dataset/house-prices-advanced-regression-techniques/test.csv")
    return train, test

def preprocess_data(data):
    none_cols = [
        "PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
        "GarageType","GarageFinish","GarageQual","GarageCond",
        "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"
    ]
    
    for col in none_cols:
        data[col] = data[col].fillna("None")
    
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
    
    cat_cols = data.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        data[col] = data[col].fillna(None)
    
    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
    
    data["TotalSF"] = (
        data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    )
    
    data["HasGarage"] = (data["GarageArea"] > 0).astype(int)
    data["HasPool"] = (data["PoolArea"] > 0).astype(int)
    
    qual_map = {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    }
    
    qual_cols = [
        "ExterQual","ExterCond","BsmtQual","BsmtCond",
        "HeatingQC","KitchenQual","FireplaceQu","GarageQual",
        "GarageCond","PoolQC"
    ]
    
    for col in qual_cols:
        data[col] = data[col].map(qual_map).fillna(0).astype(int)
    
    data["MSSubClass"] = data["MSSubClass"].astype(str)
    
    data = pd.get_dummies(data)
    
    return data

def model_train(X, y, data):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def model_predict(model, X):
    preds = model.predict(X)
    return np.expm1(preds)

def save_predictions(ids, preds):
    submission = pd.DataFrame({"Id": ids, "SalePrice": preds})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    train, test = load_data()
    train_ids = train["Id"]
    test_ids = test["Id"]
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    ntrain = train.shape[0]
    ntest = test.shape[0]
    
    y = train["SalePrice"]
    
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop("SalePrice", axis=1, inplace=True)
    all_data = preprocess_data(all_data)
    
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    
    model = model_train(train, y, all_data)
    preds = model_predict(model, test)
    save_predictions(test_ids, preds)
    print("Finished!")
    