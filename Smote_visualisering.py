import pandas as pd
import datetime as dt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



data_test = pd.read_csv(r"/Users/jens-jakobskotingerslev/Desktop/P1/data/fraudTest.csv")

data = pd.read_csv(r"/Users/jens-jakobskotingerslev/Desktop/P1/fraudTrain.csv")

 
def preprocessing(data):
    data['age'] = dt.date.today().year - pd.to_datetime(data['dob']).dt.year
    data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['day'] = pd.to_datetime(data['trans_date_trans_time']).dt.dayofweek
    data['month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month
    data = data[['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'is_fraud', 'age', 'hour', 'day', 'month', 'gender']]
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('is_fraud', axis='columns').values
    y = data['is_fraud'].values



    return data




def remove_highly_correlated_features(data, threshold=0.8):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping highly correlated columns: {to_drop}")
    data = data.drop(to_drop, axis=1)
    
    
    return data




train = preprocessing(data)
test = preprocessing(data_test)

# Ensure that non-numeric columns are encoded or processed as required here

y_train = train['is_fraud']
y_test = test['is_fraud']
X_train = train.drop('is_fraud', axis=1)
X_test = test.drop('is_fraud', axis=1)
print(X_train.dtypes)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Apply the same scaling to the test set
X_test_scaled = scaler.transform(X_test)

# Now apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_train_resampled_pca = pca.transform(X_train_resampled)

# Plot original data
plt.figure(figsize=(12, 5))

# Plot original scaled data
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], label="Not Fraud", alpha=0.5, linewidth=0.05)
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], label="Fraud", alpha=0.5, color="red", linewidth=0.05)
plt.title('Original Scaled Data')
plt.legend()

# Plot SMOTE resampled data
plt.subplot(1, 2, 2)
plt.scatter(X_train_resampled_pca[y_train_resampled == 0, 0], X_train_resampled_pca[y_train_resampled == 0, 1], label="Not Fraud", alpha=0.5, linewidth=0.05)
plt.scatter(X_train_resampled_pca[y_train_resampled == 1, 0], X_train_resampled_pca[y_train_resampled == 1, 1], label="Fraud", alpha=0.5, color="red", linewidth=0.05)
plt.title('SMOTE Data')
plt.legend()

plt.tight_layout()
plt.show()
