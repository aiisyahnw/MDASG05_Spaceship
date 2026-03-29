import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def winsorize_features(X):
    X = pd.DataFrame(X).copy()
    
    upper_limit = X.quantile(0.95)
    return X.clip(upper=upper_limit, axis=1)

def preprocess():
    os.makedirs("artifacts", exist_ok=True)
    
    df = pd.read_csv("ingested/spaceship_train.csv")
    
    df = df.drop_duplicates()

    kolom_angka = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    kolom_teks = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    
    X = df[kolom_angka + kolom_teks]
    y = df['Transported'].astype(int)

    resep_angka = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorize', FunctionTransformer(winsorize_features)), 
        ('scaler', StandardScaler())
    ])

    resep_teks = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', resep_angka, kolom_angka),
            ('cat', resep_teks, kolom_teks)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test), preprocessor

if __name__ == "__main__":
    preprocess()