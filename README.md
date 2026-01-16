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

## 3. Data Cleaning

Pertama, lakukan pengecekan missing values
```python
print("\n==== Missing Values ====")
print(df.isnull().sum())

plt.figure(figsize=(10,10))
sns.heatmap(df[['Row ID','Order Date','Date Key','Contact Name','Country','City','Region','Subregion','Customer','Customer ID','Industry','Segment','Product','Sales','Quantity','Discount','Profit']].isna())
```

Kemudian, cari duplicate rows.
```python
print("\n==== Duplicate Rows ====")
print(f"Duplicates: {df.duplicated().sum()}")
```


Terakhir lakukan feature engineering untuk mencari profit margin
```python
negative_sales = df[df['Sales'] < 0]
if not negative_sales.empty:
    print(f"Warning: Removed {len(negative_sales)} rows with Negative Sales")
    df = df[df['Sales'] >= 0]

    
df['Profit Margin'] = (df['Profit'] / df['Sales'] * 100)

# Display cleaned dataset
print(f"Cleaned Dataset Shape: {df.shape}")
display(df.head())
display(df.tail())
```

---

## 4. Data Analysis

Setelah melakukan cleaning dan feature engineering, analisa dapat dilakukan. 

Analisa pertama yang dilakukan adalah mencari transaksi yang tidak menguntungkan.

```python
loss_df = df[df['Profit'] < 0]
profit_df = df[df['Profit'] >= 0]

loss_count = len(loss_df)
total_count = len(df)
loss_persen = (loss_count / total_count) * 100

print(f"Total Transaksi: {total_count}")
print(f"Transaksi yang tidak menguntungakan: {loss_count} ({loss_persen:.2f}%)")
print(f"Total kerugian: {loss_df['Profit'].sum():.2f}")
```

Selanjutnya lakukan analisa region dan subregion. Tujuannya adalah agar informasi profit per region dan sub-region dapat dilihat.

```python
region_analysis = df.groupby(['Region', 'Subregion']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).reset_index()

#Rasio profit untuk region
region_analysis['Profit Ratio'] = region_analysis['Profit'] / region_analysis['Sales']

#Sort dari profit terendah
region_analysis = region_analysis.sort_values(by='Profit', ascending = True)

display(region_analysis)

#Visualisasi
plt.figure(figsize=(12, 6))
sns.barplot(data=region_analysis, x='Subregion', y='Profit', hue='Region', palette='viridis')
plt.title('Total Profit by Subregion')
plt.axhline(0, color='red', linestyle='--') # Garis 0 profit
plt.xticks(rotation=45)
plt.show()
```

Selanjutnya analisa produk. Analisa ini dilakukan agar pengguna bisa mendapatkan informasi terkait produk mana yang mengalami kerugian, informasi ini juga ditambah dengan pesebaran diskon dan profit.

```python
# Analisa Produk
product_analysis = df.groupby(['Product']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Discount': 'mean'
}).reset_index()

#Filter untuk produk dengan total profit negatif
product_negative = product_analysis[product_analysis['Profit'] < 0].sort_values('Profit')

print("Produk yang mengalami profit loss:")
print(product_negative)

# Visualisasi Discount
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_analysis, x='Sales', y='Profit', size='Discount', hue='Discount', sizes=(20, 200), palette='coolwarm')
plt.title('Product Performance: Sales vs Profit (Color/Size = Avg Discount)')
plt.axhline(0, color='black', linestyle='--')
plt.show()

# Visualisasi total profit by product
product_analysis = product_analysis.sort_values('Profit', ascending=True)
product_analysis['Profit Status'] = product_analysis['Profit'].apply(lambda x: 'Profit' if x >= 0 else 'Loss')

plt.figure(figsize=(12, 8))
sns.barplot(data=product_analysis, x='Profit', y='Product', hue='Profit Status', dodge=False, palette={'Profit': 'green', 'Loss': 'red'})
plt.title('Total Profit by Product')
plt.xlabel('Total Profit')
plt.ylabel('Product')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.legend(title='Status')
plt.show()
```

Setelah melakukan analisa produk, lakukan hipotesis testing untuk menguji apakah discount rate yang tinggi berpengaruh dengan profit

```python
# ==== Hypothesis testing ====


# Pembuatan threshold menjadi 2 kelompok
# kelompok A: High Discount
# kelompok B: Low Discount

# H0: Mean dari profit margin dari transaksi High Discount sama dengan transaksi Low Discount
# H1: Mean dari profit margin dari transaksi High Discount lebih rendah transaksi Low Discount
high_discount = df[df['Discount'] > 0.2]['Profit Margin']
low_discount = df[df['Discount'] <= 0.2]['Profit Margin']

t_stat, p_val = stats.ttest_ind(high_discount, low_discount, equal_var=False, alternative='less')

print("==== Hasil Hypothesis Testing ====")
print(f"Mean proft margin untuk diskon tinggi: {high_discount.mean()}")
print(f"Mean proft margin untuk diskon rendah: {low_discount.mean()}")
print(f"T-statisik: {t_stat:.4f}")
print(f"P-Value: {p_val:.4e}")

alpha = 0.05
if p_val < alpha:
    print("\nTolak H0.")
    print("Ada cukup bukti yang menyatakan bahwa penjualan dengan diskon tinggi(>20%) dapat mengakibatkan profit margin rendah")
else:
    print("\nGagal menolak H0")
    print("Tidak cukup bukti untuk menyatakan penjualan dengan diskon tinggi dapat engakibatkan profit margin rendah")
```

Kemudian lakukan analisa customer untuk mencari customer mana yang memiliki profit margin rendah

```python
df_copy = df.drop_duplicates(inplace=True)
df_copy = df[df['Sales'] >= 0]

# Customer Analysis

# Customer Grouping
cust_analys = df_copy.groupby(['Customer ID', 'Customer', 'Segment']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Discount': 'mean',
    'Quantity': 'sum',
    'Order ID': 'nunique'
}).reset_index()

# Customer Profit Margin
cust_analys['Profit Margin'] = (cust_analys['Profit'] / cust_analys['Sales'] * 100)

# Customer with negative profit
unprofit_cust = cust_analys[cust_analys['Profit'] < 0].sort_values(by='Sales', ascending=False)

print("\n==== Top 10 customer with negative profits ====")
display(unprofit_cust.head(10))
```

Terkahir, lakukan analisa segmen dan correlation testing

```python
# Segment Analysis

segment_analysis = cust_analys.groupby('Segment').agg({
    'Profit': 'sum',
    'Sales': 'sum',
    'Discount': 'mean',
    'Customer ID': 'count'
}).reset_index()
segment_analysis['Profit Margin'] = (segment_analysis['Profit']/segment_analysis['Sales'] * 100)
print("\n==== Profitability by Segment ====")
display(segment_analysis)

# Correlation Analysis
corr = cust_analys[['Sales', 'Profit', 'Discount','Profit Margin']].corr()
print("\n ==== Correlation ====")
print(corr)


# Visualization: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Customer Performance Metrics')
plt.show()

# Hypothesis Testing
# H0: Ada korelasi antara discount dengan profit margin customer
# H1: Tidak ada hubungan antara discount dengan profit margin customer


cust_analys_clean = cust_analys.dropna(subset=['Discount', 'Profit Margin'])

corr_coef, p_value = stats.pearsonr(cust_analys_clean['Discount'], cust_analys_clean['Profit Margin'])

print("\n==== Pearson Correlation Test ====")
print(f"Corr Coefficient: {corr_coef:.4f}")
print(f"P-value: {p_value:.4e}")

alpha = 0.05
if p_value < alpha:
    print("Tolak H0: Ada bukti statistik bahwa terdapat korelasi antara diskon dan profit margin")
else:
    print("Gagal tolak H0: Tidak terdapat bukti untuk menyatakan bahwa terdapat korelasi antara diskon dan profit margin")
```

