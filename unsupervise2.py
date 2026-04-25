import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# -----------------------------
# 1) อ่านไฟล์ข้อมูลที่ Clean แล้ว
# -----------------------------
df = pd.read_csv('cleansing_water_data.csv')

print("ข้อมูลทั้งหมด:", df.shape)
print(df.head())


# -----------------------------
# 2) เลือกเฉพาะคนที่ใช้ Cleansing Water
# -----------------------------
df_clean = df[df['use_cleansing_water'] == 'ใช้'].copy()

print("จำนวนข้อมูลทั้งหมด:", df.shape[0])
print("จำนวนคนที่ใช้ Cleansing Water:", df_clean.shape[0])


# -----------------------------
# 3) เลือกเฉพาะ Feature ที่จำเป็น
# ใช้เฉพาะ factor_* เพื่อแบ่งกลุ่มตามปัจจัยการเลือกซื้อ
# -----------------------------
factor_cols = [col for col in df_clean.columns if col.startswith('factor_')]

print("Factor ที่ใช้ทำ Clustering:")
print(factor_cols)


# -----------------------------
# 4) จัดการค่าว่าง
# -----------------------------
for col in factor_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())


# -----------------------------
# 5) สร้าง X สำหรับทำ Unsupervised
# ไม่มี y เพราะ Clustering ไม่ใช้ target
# -----------------------------
X = df_clean[factor_cols]

print("Shape ของ X:", X.shape)
print(X.head())


# -----------------------------
# 6) Scaling ข้อมูล
# K-Means ต้อง scale เพราะใช้ระยะห่างในการแบ่งกลุ่ม
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# 7) ลดมิติข้อมูลด้วย PCA
# ช่วยลด noise และทำให้ cluster ไม่กระจายเกินไป
# -----------------------------
pca_model = PCA(n_components=5)
X_reduced = pca_model.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca_model.explained_variance_ratio_)

print("รวมข้อมูลที่ PCA อธิบายได้:", pca_model.explained_variance_ratio_.sum())


# -----------------------------
# 8) หา k ด้วย Elbow Method
# -----------------------------
inertia = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_reduced)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


# -----------------------------
# 9) หา k ด้วย Silhouette Score
# -----------------------------
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.show()

for k, score in zip(K_range, silhouette_scores):
    print(f'k = {k}, Silhouette Score = {score:.4f}')


# -----------------------------
# 10) เลือกจำนวน Cluster
# เริ่มจาก 3 เพราะเหมาะกับการแบ่งกลุ่มลูกค้าทางการตลาด
# ถ้า Silhouette ของ k=2 หรือ k=4 ดีกว่า สามารถเปลี่ยนได้
# -----------------------------
best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_reduced)


# -----------------------------
# 11) ดูจำนวนคนในแต่ละ Cluster
# -----------------------------
print("จำนวนคนในแต่ละ Cluster:")
print(df_clean['cluster'].value_counts().sort_index())

plt.figure(figsize=(6, 4))
sns.countplot(x='cluster', data=df_clean)
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()


# -----------------------------
# 12) วาดกราฟ PCA 2 มิติ
# ใช้เพื่อดูภาพรวมการกระจายของ Cluster
# -----------------------------
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

df_clean['pca1'] = X_pca_2d[:, 0]
df_clean['pca2'] = X_pca_2d[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    data=df_clean,
    s=80
)

plt.title('Customer Segmentation using K-Means and PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()


# -----------------------------
# 13) วิเคราะห์ค่าเฉลี่ย Factor ของแต่ละ Cluster
# ตรงนี้คือส่วนสำคัญในการตีความ Insight
# -----------------------------
cluster_profile = df_clean.groupby('cluster')[factor_cols].mean()

print("ค่าเฉลี่ย Factor ของแต่ละ Cluster:")
print(cluster_profile)


plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile, annot=True, cmap='Blues', fmt='.2f')
plt.title('Average Purchase Factors by Cluster')
plt.xlabel('Factors')
plt.ylabel('Cluster')
plt.show()


# -----------------------------
# 14) แสดง Top Factors ของแต่ละ Cluster
# -----------------------------
for cluster in sorted(df_clean['cluster'].unique()):
    print("=" * 60)
    print(f"Cluster {cluster}")
    print("จำนวนคน:", len(df_clean[df_clean['cluster'] == cluster]))

    print("\nปัจจัยที่ให้ความสำคัญสูงสุด:")
    print(cluster_profile.loc[cluster].sort_values(ascending=False).head(5))

    print("\nปัจจัยที่ให้ความสำคัญต่ำสุด:")
    print(cluster_profile.loc[cluster].sort_values(ascending=True).head(3))


# -----------------------------
# 15) Export ไฟล์ผลลัพธ์
# -----------------------------
df_clean.to_csv('cleansing_water_cluster_result.csv', index=False)

print("Export ไฟล์เรียบร้อย: cleansing_water_cluster_result.csv")