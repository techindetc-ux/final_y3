# ===============================
# 1) Import Library
# ===============================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

font_path = "C:/Windows/Fonts/tahoma.ttf"
thai_font = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = thai_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 2) อ่านไฟล์ข้อมูล
# ===============================
df = pd.read_csv('cleansing_water_data.csv')

df_clean = df[df['use_cleansing_water'] == 'ใช้'].copy()

print("--- จำนวนข้อมูลทั้งหมด ---")
print(df.shape)

print("\n--- จำนวนคนที่ใช้ Cleansing Water ---")
print(df_clean.shape)


# ===============================
# 3) เลือกเฉพาะปัจจัยในการเลือกซื้อ
# ===============================
factor_cols = [c for c in df_clean.columns if c.startswith('factor_')]

print("\n--- Factor ที่ใช้ทำ Clustering ---")
print(factor_cols)


# ===============================
# 4) จัดการค่าว่าง
# ===============================
for col in factor_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())


# ===============================
# 5) สร้าง Feature สำหรับ Clustering
# ===============================
X = df_clean[factor_cols].copy()

# แปลงคะแนนเป็น pattern การเลือกซื้อ
X_pattern = X.div(X.sum(axis=1), axis=0)
X_pattern = X_pattern.fillna(0)

# ผสมคะแนนดิบกับ pattern เพื่อให้ cluster ชัดขึ้น
X_final = (X * 0.5) + (X_pattern * 0.5)


# ===============================
# 6) Scaling ข้อมูล
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)


# ===============================
# 7) ตรวจสอบ Silhouette Score
# ===============================
K_range = range(2, 7)
silhouette_scores = []

for k in K_range:
    temp_kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=100,
        max_iter=500
    )
    
    temp_labels = temp_kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, temp_labels)
    silhouette_scores.append(score)

print("\n--- Silhouette Score ของแต่ละ k ---")
for k, score in zip(K_range, silhouette_scores):
    print(f"k = {k}, Silhouette Score = {score:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score เพื่อดูจำนวน Cluster ที่เหมาะสม')
plt.xlabel('จำนวน Cluster')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()


# ===============================
# 8) สร้างโมเดล K-Means จำนวน 3 กลุ่ม
# ===============================
best_k = 3

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=100,
    max_iter=500
)

df_clean['cluster'] = kmeans.fit_predict(X_scaled)

print("\n--- จำนวนคนในแต่ละ Cluster ---")
print(df_clean['cluster'].value_counts().sort_index())


# ===============================
# 9) ลดมิติด้วย PCA เพื่อวาดกราฟ
# ===============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n--- PCA อธิบายข้อมูลได้รวม ---")
print(pca.explained_variance_ratio_.sum())

df_clean['pca1'] = X_pca[:, 0]
df_clean['pca2'] = X_pca[:, 1]


# ===============================
# 10) กราฟแสดง Cluster
# ===============================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    data=df_clean,
    palette='Set1',
    s=90,
    edgecolor='black'
)

plt.title('การแบ่งกลุ่มลูกค้า Cleansing Water ด้วย K-Means จำนวน 3 กลุ่ม')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


# ===============================
# 11) กราฟจำนวนคนในแต่ละ Cluster
# ===============================
plt.figure(figsize=(6, 4))
sns.countplot(x='cluster', data=df_clean, palette='Set1')
plt.title('จำนวนคนในแต่ละ Cluster')
plt.xlabel('Cluster')
plt.ylabel('จำนวนคน')
plt.tight_layout()
plt.show()


# ===============================
# 12) วิเคราะห์ค่าเฉลี่ย Factor ของแต่ละ Cluster
# ===============================
cluster_profile = df_clean.groupby('cluster')[factor_cols].mean()

print("\n--- ค่าเฉลี่ยปัจจัยในการเลือกซื้อของแต่ละ Cluster ---")
print(cluster_profile)


# ===============================
# 13) Heatmap วิเคราะห์ Cluster
# ===============================
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('ค่าเฉลี่ยปัจจัยในการเลือกซื้อของแต่ละ Cluster')
plt.xlabel('ปัจจัยในการเลือกซื้อ')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()


# ===============================
# 14) สรุปลักษณะเด่นของแต่ละ Cluster
# ===============================
print("\n--- สรุปลักษณะเด่นของแต่ละ Cluster ---")

for cluster in sorted(df_clean['cluster'].unique()):
    print("=" * 60)
    print(f"Cluster {cluster}")
    print("จำนวนคน:", len(df_clean[df_clean['cluster'] == cluster]))

    print("\nปัจจัยที่ให้ความสำคัญสูงสุด:")
    print(cluster_profile.loc[cluster].sort_values(ascending=False).head(5))

    print("\nปัจจัยที่ให้ความสำคัญต่ำสุด:")
    print(cluster_profile.loc[cluster].sort_values(ascending=True).head(3))


# ===============================
# 15) ตั้งชื่อกลุ่มเบื้องต้น
# แก้ได้หลังดูค่า cluster_profile จริง
# ===============================
cluster_names = {
    0: 'กลุ่มเน้นความอ่อนโยนและไม่ระคายเคือง',
    1: 'กลุ่มเน้นประสิทธิภาพการทำความสะอาด',
    2: 'กลุ่มเน้นการดูแลปัญหาผิวและสิว'
}

df_clean['cluster_name'] = df_clean['cluster'].map(cluster_names)

print("\n--- ตัวอย่างข้อมูลพร้อมชื่อ Cluster ---")
print(df_clean[['cluster', 'cluster_name']].head())


# ===============================
# 16) Export ผลลัพธ์
# ===============================
# df_clean.to_csv('cleansing_water_cluster_3groups_result.csv', index=False)

# print("\nบันทึกไฟล์เรียบร้อย: cleansing_water_cluster_3groups_result.csv")