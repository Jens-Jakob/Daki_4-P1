import pandas as pd
import datetime as dt
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, SMOTENC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction import FeatureHasher


data_train = pd.read_csv(r"C:\Users\Victor Steinrud\Downloads\fraudTrain.csv")
data_test = pd.read_csv(r"C:\Users\Victor Steinrud\Downloads\fraudTest.csv")

def preprocessing(data):
    data['age'] = dt.date.today().year - pd.to_datetime(data['dob']).dt.year
    data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['day'] = pd.to_datetime(data['trans_date_trans_time']).dt.dayofweek
    data['month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month
    data = data[['category', 'gender', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'is_fraud', 'age', 'hour', 'day', 'month']]
    data = pd.get_dummies(data, drop_first=True)
    data = remove_highly_correlated_features(data=data)
    X = data.drop('is_fraud', axis='columns').values
    y = data['is_fraud'].values
    return data, X, y

def remove_highly_correlated_features(data, threshold=0.8):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping highly correlated columns: {to_drop}")
    data = data.drop(to_drop, axis=1)
    
    return data

def feature_hash(data, columns, n_features=50):
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    for col in columns:
        # Ensure column is of string type
        data[col] = data[col].astype(str)
        # Convert column to a format suitable for FeatureHasher
        transformed_data = data[col].apply(lambda x: [x])
        hashed_features = hasher.transform(transformed_data)
        hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f"{col}_hash_{i}" for i in range(n_features)])
        data = pd.concat([data, hashed_df], axis=1).drop(col, axis=1)
    return data

columns_to_hash = ['merchant', 'first', 'last', 'job', 'state']
data_train = feature_hash(data_train, columns_to_hash)
data_test = feature_hash(data_test, columns_to_hash)

train, X_train, y_train = preprocessing(data_train)
test, X_test, y_test = preprocessing(data_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from sklearn.utils import shuffle
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)
import xgboost as xgb

xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

xgb_model.fit(X_train_resampled, y_train_resampled)

y_pred_xgb = xgb_model.predict(X_test)

print(classification_report(y_test, y_pred_xgb))
