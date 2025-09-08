import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from joblib import dump

def train_lgbm_model(df_final: pd.DataFrame):
    os.makedirs("models", exist_ok=True)

    y = df_final["Target"].astype(int)
    X = df_final.drop(columns=["Target"]).copy()

    # Train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1) Önce tüm object kolonları category'ye çevir (LightGBM object sevmez)
    obj_cols_train = X_train.select_dtypes(include=["object"]).columns.tolist()
    obj_cols_valid = X_valid.select_dtypes(include=["object"]).columns.tolist()
    for c in set(obj_cols_train).union(obj_cols_valid):
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    # 2) Kritikte olmasını istediğimiz kategorikler (varsa)
    must_cat = [
        "CustomerId", "ProductId",
        "product_category", "product_subcategory",
        "last_bought_category", "Sku", "BrandId"
    ]
    for c in must_cat:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_valid[c] = X_valid[c].astype("category")

    # 3) Model
    print("Model eğitiliyor...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    dump(model, "models/lgbm_model.pkl")
    print("Model başarıyla eğitildi ve 'models/lgbm_model.pkl' olarak kaydedildi.")
    return model, X_train
