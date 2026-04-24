import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


df = pd.read_csv('cleansing_water_data.csv')

print(df.shape)
df.head()


#---------------เลือกเฉพาะคนที่ใช้ Cleansing Water
df_clean = df[df['use_cleansing_water'] == 'ใช้'].copy()

print("จำนวนข้อมูลทั้งหมด:", df.shape[0])
print("จำนวนคนที่ใช้ Cleansing Water:", df_clean.shape[0])


#----------------เลือกเฉพาะคนที่ใช้ Cleansing Water
df_clean = df[df['use_cleansing_water'] == 'ใช้'].copy()

print("จำนวนข้อมูลทั้งหมด:", df.shape[0])
print("จำนวนคนที่ใช้ Cleansing Water:", df_clean.shape[0])


#----------------เลือกคอลัมน์ที่จะใช้ทำ Clustering
factor_cols = [col for col in df_clean.columns if col.startswith('factor_')]

print(factor_cols)


#-----------------จัดการค่าว่าง
for col in factor_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

df_clean['concerns'] = df_clean['concerns'].fillna('')
df_clean['skin_type'] = df_clean['skin_type'].fillna('ไม่ระบุ')
df_clean['age'] = df_clean['age'].fillna('ไม่ระบุ')
df_clean['monthly_income'] = df_clean['monthly_income'].fillna('ไม่ระบุ')


#---------------แปลงข้อมูลตัวอักษรเป็นตัวเลข
def get_dummies_multiselect(series, prefix):
    return series.str.get_dummies(sep=',').add_prefix(f'{prefix}_')

concerns_df = get_dummies_multiselect(df_clean['concerns'], 'concern')
skin_type_df = pd.get_dummies(df_clean['skin_type'], prefix='skin')

le_age = LabelEncoder()
le_income = LabelEncoder()

df_clean['age_encoded'] = le_age.fit_transform(df_clean['age'])
df_clean['income_encoded'] = le_income.fit_transform(df_clean['monthly_income'])


#-------------รวม Features
X = pd.concat([
    df_clean[factor_cols],
    concerns_df,
    skin_type_df,
    df_clean[['age_encoded', 'income_encoded']]
], axis=1)

print(X.shape)
X.head()


#----------Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#------------หา k ด้วย Elbow Method
inertia = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


#----------------หา k ด้วย Silhouette Score
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.show()

for k, score in zip(K_range, silhouette_scores):
    print(f'k = {k}, Silhouette Score = {score:.4f}')
    
    
    
#--------------สร้าง K-Means Model
best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

df_clean[['age', 'monthly_income', 'skin_type', 'cluster']].head()


#---------------ดูจำนวนคนแต่ละกลุ่ม
print(df_clean['cluster'].value_counts().sort_index())

plt.figure(figsize=(6, 4))
sns.countplot(x='cluster', data=df_clean)
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()


#--------------วาดกราฟ PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean['pca1'] = X_pca[:, 0]
df_clean['pca2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    data=df_clean,
    s=80
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


#---------------วิเคราะห์ค่าเฉลี่ย Factor ของแต่ละ Cluster
cluster_profile = df_clean.groupby('cluster')[factor_cols].mean()
cluster_profile


plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile, annot=True, cmap='Blues', fmt='.2f')
plt.title('Average Purchase Factors by Cluster')
plt.xlabel('Factors')
plt.ylabel('Cluster')
plt.show()


#-----------------ดูลักษณะของแต่ละกลุ่ม
for cluster in sorted(df_clean['cluster'].unique()):
    print("=" * 50)
    print(f"Cluster {cluster}")
    print("จำนวนคน:", len(df_clean[df_clean['cluster'] == cluster]))

    print("\nประเภทผิวที่เจอบ่อย:")
    print(df_clean[df_clean['cluster'] == cluster]['skin_type'].value_counts().head(3))

    print("\nช่วงอายุที่เจอบ่อย:")
    print(df_clean[df_clean['cluster'] == cluster]['age'].value_counts().head(3))

    print("\nปัจจัยที่ให้ความสำคัญสูงสุด:")
    print(cluster_profile.loc[cluster].sort_values(ascending=False).head(5))
    
    #Export ไฟล์ผลลัพธ์
    #df_clean.to_csv('cleansing_water_cluster_result.csv', index=False)
