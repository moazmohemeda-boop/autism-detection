import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Autism_Data.arff")
print(df.head(20))
print(df.info())
df.replace("?", np.nan, inplace=True)
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())
print(df.shape)
cols_to_drop = ['jundice', 'austim', 'contry_of_res', 'age_desc', 'relation', 'ethnicity']
cols_present = [c for c in cols_to_drop if c in df.columns]
if cols_present:
    df.drop(cols_present, axis=1, inplace=True)
print(df.describe().T)
pd.set_option('future.no_silent_downcasting', True)
df.replace({'yes': 1, 'no': 0}, inplace=True)
df.replace({'m': 1, 'f': 0}, inplace=True)
X = df.drop('Class/ASD', axis=1)
Y = df['Class/ASD']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
print("KNN Train Score:", knn_model.score(X_train, Y_train))
print("KNN Test  Score:", knn_model.score(X_test,  Y_test))
Y_pred_knn = knn_model.predict(X_test)
print("\n--- KNN Results ---")
print("Accuracy:", accuracy_score(Y_test, Y_pred_knn))
print(classification_report(Y_test, Y_pred_knn))
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, Y_train)
Y_pred_lr = lr_model.predict(X_test)
print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(Y_test, Y_pred_lr))
print(classification_report(Y_test, Y_pred_lr))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(Y_test, Y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred_lr), annot=True, fmt='d', cmap='Greens')
plt.title('Logistic Regression Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
