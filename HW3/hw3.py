from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
import os
path = 'D:\\essay\\OneDrive - National ChengChi University\\桌面\\113-01\\資料採掘\hw3'
os.chdir(path)
data = pd.read_csv("genres_v2.csv")
print(data.head())
#
data.info()
data.isnull().sum()
#drop unnecessary column
data.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'song_name', 'Unnamed: 0', 'title'], inplace=True)
# drop duplicates
data = data.drop_duplicates()
data.isnull().sum()

data.info()
#One-Hot Encoding (將類別變數轉成數值化)
time_signature = pd.get_dummies(data['time_signature'], prefix='time_signature')
mode = pd.get_dummies(data['mode'], prefix='mode')
data = pd.concat([data, time_signature],axis=1)
data = pd.concat([data, mode],axis=1)

#eda
data['genre'].unique()

ignore = ['genre','time_signature_1','time_signature_3','time_signature_4', 'time_signature_5','mode_0','mode_1']

for col in list(columns for columns in data.columns.tolist() if columns not in ignore)  :
    plt.hist(data[col])
    plt.title(col)
    plt.show()


data['duration_min'] = data['duration_ms']/60000
data.drop(columns=['duration_ms'],inplace=True)

# 提取目標變數和特徵
X = data.drop(columns=['genre'])
y = data['genre']

# 處理資料不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
from collections import Counter
Counter(y_resampled)
Counter(y)
# 分割訓練和測試資料
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
'''
# 建立 Random Forest 模型
'''

rf = RandomForestClassifier(random_state=42, 
                            class_weight='balanced',
                            n_estimators=100,
                            max_depth=30)#The number of trees in the forest
rf.fit(X_train, y_train)
#train result
y_pred = rf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)

print(f"Train Accuracy: {accuracy:.4f}")


# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores}")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# 顯示 test result
y_pred = rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
# 繪製混淆矩陣
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# 特徵重要性
importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1] #由大到小

plt.figure(figsize=(12, 8))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# 提高準確率的建議
# 1. 調整模型參數（例如使用 GridSearchCV 或 RandomizedSearchCV 尋找最佳參數）。
# 2. 嘗試不同的處理不平衡資料的方法，例如 RandomUnderSampler 或其他。
# 3. 測試其他模型（如 Gradient Boosting、XGBoost 或 LightGBM）。
# 4. 添加新特徵，或嘗試進一步特徵工程。

'''
### SVC ###
'''
from sklearn import svm
# 建立 SVM 模型
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("genres_v2.csv")

#drop unnecessary column
data.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'song_name', 'Unnamed: 0', 'title'], inplace=True)
data = data.drop_duplicates()
data.isnull().sum()

data.info()
data['duration_min'] = data['duration_ms']/60000
data.drop(columns = ['duration_ms'],inplace = True)
# Identify numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Scale only the numerical columns
data[numerical_cols] = min_max_scaler.fit_transform(data[numerical_cols])


#One-Hot Encoding (將類別變數轉成數值化)
time_signature = pd.get_dummies(data['time_signature'], prefix='time_signature')
mode = pd.get_dummies(data['mode'], prefix='mode')
data = pd.concat([data, time_signature],axis=1)
data = pd.concat([data, mode],axis=1)

# 提取目標變數和特徵
X = data.drop(columns=['genre'])
y = data['genre']

# 處理資料不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 分割訓練和測試資料
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a linear SVM model
svm_model = svm.SVC(kernel='linear', C=1000)
# cv交叉驗證
k = 3
cv_score = cross_val_score(svm_model, X_train, y_train, cv=k)
print('Cross_val Scores: ', cv_score)
print("Train Accuracy(average):", cv_score.mean()) 
# test 
svm = svm_model.fit(X_train,y_train)
y_pred = svm.predict(X_test)
# accuracy
score_accuracy_svm = accuracy_score(y_test, y_pred)
print("Test Accuracy:", score_accuracy_svm)

# Extract feature coefficients
feature_coefficients = svm_model.coef_

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Cross-Validation Accuracy: {cv_score}")

print(f"Cross-Validation Accuracy: {cv_score.mean():.4f}")

print("Confusion Matrix:")
print(conf_matrix)

# 顯示分類報告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 繪製混淆矩陣
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Plot feature coefficients
features = X.columns  # Assuming X is your features DataFrame
coefficients = feature_coefficients[0]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(features, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients from Linear SVM')
plt.show()


#-----------------------------------------------------------------------------
## choose top 10 features
#
df = pd.DataFrame({
    'Features':features,
    'Coefficient Value':coefficients,
    'importance':abs(coefficients)
    })
# Sort the DataFrame by the absolute value of the coefficient
top10_f = df.sort_values(by='importance', ascending=False)['Features'].head(10)
## hw2 retry
'''
kmeans
'''
from sklearn.cluster import KMeans
X_selected = X[top10_f]
numerical_cols = X_selected.select_dtypes(include=['float64', 'int64']).columns

# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Scale only the numerical columns
X_selected[numerical_cols] = min_max_scaler.fit_transform(X_selected[numerical_cols])

X_selected_kmean = X_selected

# 計算 SSE
sse = []
k_range = range(1, 15)  # 設定要測試的群集數範圍
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_selected_kmean)
    sse.append(kmeans.inertia_)  # inertia_ 是 SSE 的值

# 繪製肘部圖
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()

###  k = 15
kmeans_15 = KMeans(n_clusters=15,max_iter=100)
kmeans_15.fit(X_selected_kmean)

kmeans_15.labels_
from sklearn.metrics import rand_score, adjusted_rand_score,normalized_mutual_info_score,v_measure_score,fowlkes_mallows_score,confusion_matrix,silhouette_score
X_selected_kmean['cluster15'] = kmeans_15.labels_
X_selected_kmean['genre'] = y
group_map = X_selected_kmean.groupby('cluster15')['genre'].agg(lambda x: x.value_counts().idxmax())
X_selected_kmean['cluster15']=X_selected_kmean['cluster15'].map(group_map)
# Example labels_true and labels_pred
labels_true = y

labels_pred = X_selected_kmean['cluster15']

sklearn_rand_score = rand_score(labels_true, labels_pred) # Calculate Rand Score
sklearn_adjusted_rand_score = adjusted_rand_score(labels_true, labels_pred) # Calculate Adjusted Rand Score
sklearn_normalized_mutual_info_score = normalized_mutual_info_score(labels_true, labels_pred,average_method='geometric') # Calculate normalized_mutual_info_score
sklearn_v_measure_score = v_measure_score(labels_true, labels_pred)#average_method =‘arithmetic’
sklearn_fowlkes_mallows_score = fowlkes_mallows_score(labels_true, labels_pred)
sklearn_silhouette_score = silhouette_score(X_selected,kmeans_15.labels_)
sklearn_confusion_matrix = confusion_matrix(labels_true, labels_pred)

print("Rand Score (sklearn):", sklearn_rand_score)
print("Adjusted Rand Score (sklearn):", sklearn_adjusted_rand_score)
print("Normalized Mutual Information :", sklearn_normalized_mutual_info_score)
print("V-measure  :", sklearn_v_measure_score)
print("Fowlkes-Mallows index (FMI)  :", sklearn_fowlkes_mallows_score)
print("Silhouette Coefficient  :", sklearn_silhouette_score)
print("confusion matrix  :\n", sklearn_confusion_matrix)
'''
hierarchy
'''
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
X_selected_h = X_selected
linkage_matrix = linkage(X_selected_h, method='ward')
# Identify numerical columns
## plot
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()