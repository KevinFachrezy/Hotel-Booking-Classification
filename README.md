# Hotel-Booking-Classification
Project Hotel Booking classification adalah sebuah project machine learning dimana model akan mempredeksi kemungkinan pelanggan melakukan pembatalan reservasi.

---

# Penjelasan komponen sistem

---

## 1. Import Library

Sebelum melakukan data analysis, ada beberapa library yang harus diimport.

```python
# Library

import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from IPython.display import display

# Feature Engineering
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import category_encoders as ce

# Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, roc_auc_score

# Imbalance Dataset
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# Ignore Warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set max columns
pd.set_option('display.max_columns', None)
```

---

## 2. Import Dataset

Setelah dataset didownload, lakukan load data.

```python
df = pd.read_csv("data_hotel_booking_demand.csv")
df.head()
```

---

## 3. Data Analysis

Pertama, lakukan pengecekan missing values
```python
listItem = []
for col in df.columns :
    listItem.append([col, df[col].dtype, df[col].isna().sum(), round((df[col].isna().sum()/len(df[col])) * 100,2),
                    df[col].nunique(), list(df[col].drop_duplicates().sample(2).values)]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
dfDesc
```
Barplot missing values
```python
missingno.bar(df,color="dodgerblue", sort="ascending", figsize=(10,5), fontsize=12);
```
Matrix Missing value
```python
missingno.matrix(df)
```
note: Missing value yang ditemukan nanti akan dilakukan imputation sebelum training.

Kemudian mencari pesebaran data
```python
plt.figure(figsize = (10,5))
sns.countplot(df['market_segment'])
plt.show()
```

Histogram
```python
plt.figure(figsize=(17,12))

plt.subplot(221)
sns.histplot(data=df,x='previous_cancellations',hue='is_canceled',kde=True)
plt.title('Previous Cancellation Histogram',fontsize=20)

plt.subplot(222)
sns.histplot(data=df,x='booking_changes',hue='is_canceled',kde=True)
plt.title('Booking Changes Histogram',fontsize=20)

plt.subplot(223)
sns.histplot(data=df,x='total_of_special_requests',hue='is_canceled',kde=True)
plt.title('Special Requests Histogram',fontsize=20)

plt.subplot(224)
sns.histplot(data=df,x='days_in_waiting_list',hue='is_canceled',kde=True)
plt.title('Day in Waiting List Histogram',fontsize=20)
```

Barplot untuk pesebaran data
```python
count = 0
fig = plt.figure(figsize=(20,20))

for i in df.drop(columns=['is_canceled','country','previous_cancellations','booking_changes', 'days_in_waiting_list', 'total_of_special_requests', 'required_car_parking_spaces']).columns:
    count +=1
    ax= plt.subplot(4,2,count)
    pd.crosstab(df[i],df['is_canceled'],normalize=0).plot(kind='bar',stacked=True,ax=ax)
    fig.tight_layout()

plt.show()
```
Table untuk pesebaran data
```python
for i in df.drop(columns=['is_canceled','country','previous_cancellations','booking_changes', 'days_in_waiting_list', 'total_of_special_requests', 'required_car_parking_spaces']).columns:
    is_canceled_df = df.groupby(i)['is_canceled'].value_counts(normalize=True).unstack()
    display(is_canceled_df.sort_values(by=[1.0], ascending=False))
```
---

## 4. Data Preparation

Setelah melakukan analysis, data harus dipersiapkan untuk model training

Tahap pertama adalah handling missing value dan grouping,
* Missing value akan diimput menggunakan strategi constant dan value 'N/A'
* Grouping akan mengelompokan top 10 country yang ada di dalam dataset dan sisanya menjadi 'Other'

```python
imputer_cat = SimpleImputer(strategy='constant', fill_value='N/A')
df['country'] = imputer_cat.fit_transform(df[['country']]).ravel()

top_10_countries = df['country'].value_counts().nlargest(10).index
df['country'] = df['country'].apply(lambda x: x if x in top_10_countries else 'Other')
```

Kemudian kita melakukan encoding,
* **RobustScaler**: untuk fitur yang memiliki outliers (`previous_cancellations`, `booking_changes`).
* **MinMaxScaler**: untuk fitur yang memiliki boundary (`total_of_special_requests`, `required_car_parking_spaces`).
* **Log Transformation**: untuk menghandle data yang skewed (`days_in_waiting_list`).
* **OneHotEncoder**: untuk data kategorikal
```python
preprocess = ColumnTransformer(
    transformers=[
        ('num_robust', RobustScaler(), ['previous_cancellations', 'booking_changes']),
        ('num_minmax', MinMaxScaler(), ['total_of_special_requests', 'required_car_parking_spaces']),
        ('num_log', Pipeline([
            ('log', FunctionTransformer(np.log1p)), 
            ('scaler', MinMaxScaler())
        ]), ['days_in_waiting_list']),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), 
         ['country', 'market_segment', 'deposit_type', 'customer_type', 'reserved_room_type'])
    ], remainder='passthrough'
)
```

Selanjutnya lakukan data splitting untuk testing dan training

```python
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']
```

```python
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=2021)
```

Kemudian kita akan melakukan model initiation. Dalam project ini ada 5 model yang digunakan
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. XGBoost

```python
logreg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
```

---

## 5. Model training
Tahap pertama dalam training adalah melakukan benchmarking untuk megetahui model terbaik untuk klasifikasi. Ada 2 benchmarking yang akan dilakukan, K-fold dan Test Data benchmarking.

K-Fold Benchmark
```python
models = [logreg,knn,dt,rf,xgb]
score=[]
rata=[]
std=[]
for name in models:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocess), 
        ('model', name)
        ])
    cv_scores = cross_val_score(clf_pipeline, X_train, y_train, cv=skf, scoring='roc_auc')
    score.append(cv_scores)
    rata.append(cv_scores.mean())
    std.append(cv_scores.std())
    
pd.DataFrame({'model':['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGB'],'mean roc_auc':rata,'sdev':std}).set_index('model').sort_values(by='mean roc_auc',ascending=False)
```
Test Data Benchmark
```python
score_roc_auc = []

def y_pred_func(i):
    estimator=Pipeline([
        ('preprocess',preprocess),
        ('model',i)])
    X_train,X_test
    
    estimator.fit(X_train,y_train)
    return(estimator,estimator.predict(X_test),X_test)

for i,j in zip(models, ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGB']):
    estimator,y_pred,X_test = y_pred_func(i)
    y_predict_proba = estimator.predict_proba(X_test)[:,1]
    score_roc_auc.append(roc_auc_score(y_test,y_predict_proba))
    print(j,'\n', classification_report(y_test,y_pred))
    
pd.DataFrame({'model':['Logistic Regression','KNN', 'Decision Tree', 'Random Forest', 'XGB'],
             'roc_auc score':score_roc_auc}).set_index('model').sort_values(by='roc_auc score',ascending=False)
```
Setelah melakukan benchmarking, score tertinggi ada pada model XGB seperti image dibawah.

<img width="317" height="248" alt="image" src="https://github.com/user-attachments/assets/4069aed0-5845-445c-9481-b36c9700fcb3" />

Tahap selanjutnya adalah melakukan oversampling untuk melihat apakah performa XGB dapat ditingkatkan.

Function definition
```python
# Functions
def calc_train_error(X_train, y_train, model):
#     '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:,1])
    recall = recall_score(y_train, predictions)
    precision = precision_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }
    
    
def calc_validation_error(X_test, y_test, model):
#     '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:,1])
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
#     '''fits model and returns the in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error
```

Test oversampling with K-fold
```python
k = 10
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

data = X_train
target = y_train

train_errors_without_oversampling = []
validation_errors_without_oversampling = []

train_errors_with_oversampling = []
validation_errors_with_oversampling = []

for train_index, val_index in kfold.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    Y_train, Y_val = target.iloc[train_index], target.iloc[val_index]
    
#     print(len(X_val), (len(X_train) + len(X_val)))
    ros = RandomOverSampler()

    X_ros, Y_ros = ros.fit_resample(X_train, Y_train)

    # instantiate model
    xgb = XGBClassifier()
    estimator=Pipeline([
        ('preprocess',preprocess),
        ('model',xgb)
    ])

    #calculate errors
    train_error_without_oversampling, val_error_without_oversampling = calc_metrics(X_train, Y_train, X_val, Y_val, estimator)
    train_error_with_oversampling, val_error_with_oversampling = calc_metrics(X_ros, Y_ros, X_val, Y_val, estimator)
    
    # append to appropriate list
    train_errors_without_oversampling.append(train_error_without_oversampling)
    validation_errors_without_oversampling.append(val_error_without_oversampling)
    
    train_errors_with_oversampling.append(train_error_with_oversampling)
    validation_errors_with_oversampling.append(val_error_with_oversampling)
```

Kemudian kita akan melihat score di kedua tes
Untuk validation error tanpa oversampling
```python
for rep in validation_errors_without_oversampling :
    print(rep['report'])
```
Untuk validation error dengan oversampling
```python
for rep in validation_errors_with_oversampling :
    print(rep['report'])
```
Setelah melihat result score dari oversampling, model yang digunakan adalah XGB yang telah dioversample.


---

## 6. Hyperparameter tuning
Setelah melakukan benchmark, model XGB akan dilakukan hyperparameter menggunakan RandomizedSearch untuk mencari apakah konfigurasi setelah hyperparameter tuning lebih baik dari default model.

Instantiate model menggunakan pipeline dan RandomOverSampler
```python
xgb = XGBClassifier()
ros = RandomOverSampler()

estimator=Pipeline([
    ('oversampling', ros),
    ('preprocess',preprocess),
    ('model',xgb)
])
```
Parameter definition
```python
param_dist = {
    'model__learning_rate': uniform(0.01, 0.19),      
    'model__max_depth': randint(3, 11),               
    'model__min_child_weight': randint(1, 6),         
    'model__gamma': uniform(0.0, 0.5),                
    'model__subsample': uniform(0.6, 0.4),            
    'model__colsample_bytree': uniform(0.6, 0.4)     
}
```
Random Search definition
```python
random_search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=param_dist,
    n_iter=40,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
```

Setelah melakukan parameter definition dan Random Search definition, kita akan melakukan fitting. Di proses fitting ini kita juga akan melakukan perbandingan antara XGB sebelum hyperparameter tuning dan setelah tuning.
Fitting untuk random seach
```python
random_search.fit(X_train, y_train)
print(random_search.best_score_)
print(random_search.best_params_)
```
fitting untuk best_model menggunakan best estimator dari random search.
```python
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)
```
Fitting untuk XGB sebelum hyperparameter tuning
```python
estimator=Pipeline([
    ('preprocess',preprocess),
    ('model',xgb)
])
estimator.fit(X_train, y_train)
```

Setelah melakukan fitting, kita akan melihat result klasifikasi kedua model
ROC AUC score
```python
y_pred_default = estimator.predict(X_test)
y_pred_proba_default = estimator.predict_proba(X_test)
y_pred_tuned = best_model.predict(X_test)
y_pred_proba_tuned = best_model.predict_proba(X_test)

roc_auc_default = roc_auc_score(y_test, y_pred_proba_default[:,1])
roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned[:,1])

print('ROC AUC Score Default XGB : ', roc_auc_default)
print('ROC AUC Score Tuned XGB : ', roc_auc_tuned)
```
Classification Report
```python
report_default = classification_report(y_test, y_pred_default)
report_tuned = classification_report(y_test, y_pred_tuned)

print('Classification Report Default XGB : \n', report_default)
print('Classification Report Tuned XGB : \n', report_tuned)
```
