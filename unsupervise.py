import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# อ่านไฟล์ Excel โดยข้ามแถวแรกเหมือนโค้ด rename.py
data = pd.read_excel('BU Data from Survey Cases_final.xlsx', skiprows=1)

# เปลี่ยนชื่อคอลัมน์จากภาษาไทยเป็นภาษาอังกฤษ
rename_map = {
    'เพศ': 'sex',
    'อายุ': 'age',
    'โปรดระบุอาชีพของคุณ (อื่นๆ โปรดระบุ)': 'occupation',
    'โปรดเลือกรายได้ต่อเดือนของคุณ': 'monthly_income',
    'โปรดพิมพ์จังหวัดที่อยู่อาศัยของคุณ เช่น กทม. , ขอนแก่น, ชลบุรี': 'province',
    'โปรดเลือกประเภทผิวของคุณ': 'skin_type',
    'คุณมีความกังวล/ปัญหาผิวในเรื่องใดบ้าง (เลือกได้หลายข้อ)': 'concerns',
    'คุณเป็นสิวหรือไม่ เป็นสิวรุนแรงระดับใด': 'acne_level',
    "คุณปรึกษาหรือได้รับอิทธิพลจากใครในการเลือกสกินแคร์ 'สำหรับผิวหน้า' บ้าง (เลือกได้หลายคำตอบ)": 'consult_influencer',
    "คุณเลือกสกินแคร์ 'สำหรับผิวหน้า' อย่างไร (เลือกได้หลายข้อ)": 'skincare_face_method',
    'คุณเลือกคลีนซิ่ง เช่น Cleansing water, cleansing balm, cleansing oil อย่างไร (เลือกได้หลายข้อ)': 'cleansing_method',
    'คุณใช้คลีนซิ่งแบบน้ำ (Cleansing water) หรือไม่': 'use_cleansing_water',
    'ปัจจุบันคุณใช้คลีนซิ่งแบบใดบ้าง (หากใช้หลายแบบ เลือกได้หลายคำตอบ)': 'cleansing_types_used',
    'คุณใช้คลีนซิ่งแบบใดมากที่สุด (เลือกคำตอบเดียว)': 'cleansing_type_most_used',
    'คุณใช้คลีนซิ่ง (Cleansing water) สูตรใด (หากใช้หลายแบบ เลือกได้หลายคำตอบ)': 'cleansing_water_formula',
    'ปัจจัยใดบ้างที่ส่งผลต่อการเปลี่ยนหรือทดลองคลีนซิ่งใหม่ (อื่นๆ โปรดพิมพ์ระบุเหตุผลสั้นๆ)': 'switch_factors',
    'ปัจจุบันคุณใช้คลีนซิ่งแบรนด์ใดอยู่บ้าง (เลือกได้หลายคำตอบ, เลือกอื่นๆ โปรดระบุ)': 'brands_used',
    'ปัจจุบันคุณใช้คลีนซิ่งแบรนด์ใดบ่อยที่สุด (เลือกเพียงคำตอบเดียว)': 'brand_primary',
}

data.rename(columns=rename_map, inplace=True)

# เปลี่ยนชื่อ factor columns ให้สั้นลง
factor_rename = {}
for col in data.columns:
    if 'เช็ดเมคอัพสะอาดหมดจด' in col:
        factor_rename[col] = 'factor_deep_cleansing'
    elif 'ช่วยลดสิว' in col:
        factor_rename[col] = 'factor_acne_friendly'
    elif 'อ่อนโยนต่อผิวแพ้ง่าย' in col:
        factor_rename[col] = 'factor_sensitive_friendly'
    elif 'ผ่านการทดสอบทางการแพทย์' in col:
        factor_rename[col] = 'factor_hypoallergenic'
    elif 'ชุ่มชื้น' in col:
        factor_rename[col] = 'factor_moisturizing'
    elif 'ลดแรงเสียดสี' in col:
        factor_rename[col] = 'factor_low_friction'
    elif 'มีสารบำรุง' in col:
        factor_rename[col] = 'factor_nourishment'
    elif 'รอบดวงตา' in col:
        factor_rename[col] = 'factor_eye_friendly'
    elif 'ควบคุมความมัน' in col:
        factor_rename[col] = 'factor_oil_control'
    elif 'ไม่มีสารก่อการแพ้' in col:
        factor_rename[col] = 'factor_no_allergen'

data.rename(columns=factor_rename, inplace=True)

# ลบ Timestamp เพราะไม่ใช้ในการแบ่งกลุ่ม
data = data.drop(['Timestamp'], axis=1)



#----------------กรองเฉพาะคนที่ใช้ Cleansing Water
df_clean = data[data['use_cleansing_water'] == 'ใช้'].copy()

print("จำนวนข้อมูลทั้งหมด:", data.shape)
print("จำนวนคนที่ใช้ Cleansing Water:", df_clean.shape)



#-----------------เตรียม Feature สำหรับ Unsupervised
factor_cols = [c for c in df_clean.columns if c.startswith('factor_')]

print("Factor columns:")
print(factor_cols)


# เติมค่าว่างใน factor ด้วยค่ากลาง
for col in factor_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    
    # ฟังก์ชันแปลงคำตอบแบบเลือกได้หลายข้อให้เป็น one-hot
def get_dummies_multiselect(series, prefix):
    return series.fillna('').str.get_dummies(sep=',').add_prefix(f"{prefix}_")


# แปลง concerns เป็น one-hot
concerns_df = get_dummies_multiselect(df_clean['concerns'], 'concern')

# แปลง skin_type เป็น one-hot
skin_type_df = pd.get_dummies(df_clean['skin_type'], prefix='skin')

# แปลง age และ income เป็นตัวเลข
le_age = LabelEncoder()
le_income = LabelEncoder()

df_clean['age_encoded'] = le_age.fit_transform(df_clean['age'].astype(str))
df_clean['income_encoded'] = le_income.fit_transform(df_clean['monthly_income'].astype(str))



#-----------------รวม Feature ทั้งหมด

X = pd.concat([
    df_clean[factor_cols],
    concerns_df,
    skin_type_df,
    df_clean[['age_encoded', 'income_encoded']]
], axis=1)

print("Shape ของ X:", X.shape)
X.head()



#-------------------Scaling ข้อมูล

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



#------------------หา k ที่เหมาะสมด้วย Elbow Method

inertia = []

K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Finding Best K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()



#-----------------หา k ด้วย Silhouette Score

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Finding Best K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

for k, score in zip(K_range, silhouette_scores):
    print(f"k = {k}, Silhouette Score = {score:.4f}")
    
    
    
#--------------------เลือก k แล้วสร้าง K-Means Model

best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

df_clean[['age', 'monthly_income', 'skin_type', 'cluster']].head()


#----------------ดูจำนวนคนในแต่ละ Cluster

cluster_count = df_clean['cluster'].value_counts().sort_index()

print(cluster_count)

plt.figure(figsize=(6, 4))
sns.countplot(x='cluster', data=df_clean)
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()



#------------------วาดกราฟ PCA ให้เห็นกลุ่ม

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
    palette='Set2',
    s=80
)

plt.title('Customer Segmentation using K-Means and PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()


#-----------------วิเคราะห์ค่าเฉลี่ยของ Factor แต่ละ Cluster

cluster_profile = df_clean.groupby('cluster')[factor_cols].mean()

cluster_profile

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile, annot=True, cmap='Blues', fmt='.2f')
plt.title('Average Purchase Factors by Cluster')
plt.xlabel('Factors')
plt.ylabel('Cluster')
plt.show()


#----------------ดู Skin Type หลักของแต่ละ Cluster
for cluster in sorted(df_clean['cluster'].unique()):
    print(f"\nCluster {cluster}")
    print(df_clean[df_clean['cluster'] == cluster]['skin_type'].value_counts().head())


#---------------ดูปัญหาผิวหลักของแต่ละ Cluster

concerns_with_cluster = concerns_df.copy()
concerns_with_cluster['cluster'] = df_clean['cluster'].values

concern_profile = concerns_with_cluster.groupby('cluster').mean()

plt.figure(figsize=(12, 6))
sns.heatmap(concern_profile, annot=True, cmap='Oranges', fmt='.2f')
plt.title('Skin Concerns by Cluster')
plt.xlabel('Skin Concerns')
plt.ylabel('Cluster')
plt.show()


#-----------สรุป Cluster Profile แบบอ่านง่าย

for cluster in sorted(df_clean['cluster'].unique()):
    print("="*50)
    print(f"Cluster {cluster}")
    print("จำนวนคน:", len(df_clean[df_clean['cluster'] == cluster]))
    
    print("\nประเภทผิวที่พบมาก:")
    print(df_clean[df_clean['cluster'] == cluster]['skin_type'].value_counts().head(3))
    
    print("\nค่าเฉลี่ยปัจจัยที่สำคัญที่สุด:")
    print(cluster_profile.loc[cluster].sort_values(ascending=False).head(5))
    
    