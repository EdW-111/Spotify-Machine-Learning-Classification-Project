"""
Created on Thu May  1 13:13:32 2025

@author: eddiewang
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
from torch import nn, optim

random.seed(14562458)
rs = 14562458

filepath = "/Users/eddiewang/Desktop/F_Machine_Learning/CAPSTONE_PROJECT/musicData.csv"

#Cleaning Data
df = pd.read_csv(filepath)
print("Original shape:", df.shape)
missing_rows = df[df.isna().any(axis=1)]
print("Rows with missing values:", missing_rows.shape)
df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
numerical = ['popularity','acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','valence','tempo']
df['tempo'] = df.groupby('music_genre')['tempo'].transform(
    lambda x: x.fillna(x.median())
)

df.dropna(inplace=True)

columns_to_drop = ['instance_id', 'track_name', 'obtained_date']

df.drop(columns=columns_to_drop, inplace=True)
print("After dropping non-feature columns:", df.shape)

for i in numerical:
    print(i,"sum of NA:" , df[i].isna().sum())



COl_TYPE = ['popularity', 'acousticness', 'danceability', 'duration_ms',
            'energy', 'instrumentalness', 'liveness', 'loudness', 
            'speechiness', 'tempo', 'valence' ,'music_genre']
genre_counts = df['music_genre'].value_counts()
unique_genres = df['music_genre'].nunique()

print("Genre Counts:", genre_counts)
print("Number of Unique Genres:", unique_genres)


# Visualization
numberical = ['popularity','acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','valence']
for col in numberical:
    sns.histplot(data=df, x=col, hue='music_genre', element='step', stat='density')
    plt.title(f'{col.capitalize()} by Genre')
    plt.tight_layout()
    plt.show()
categorical = ['key', 'mode', 'music_genre']
for col in categorical:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, hue='music_genre')
    plt.title(f'{col.capitalize()} by Genre')
    plt.tight_layout()
    plt.show()
    
#Acousticness Transformation:
df['acousticness_log'] = np.log1p(df['acousticness'])
acoustic_data = df['acousticness'].dropna()
mu = acoustic_data.mean()
sigma = acoustic_data.std()
plt.figure(figsize=(10, 5))
count, bins, ignored = plt.hist(acoustic_data, bins=30, density=True, color='skyblue', alpha=0.7, edgecolor='black', label='Histogram')
x = np.linspace(min(bins), max(bins), 100)
normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))
plt.plot(x, normal_curve, color='red', linewidth=2, label='Normal Distribution')
plt.title('Distribution of Acousticness (with Normal Curve)')
plt.xlabel('Acousticness')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

plt.hist(df['acousticness_log'], bins=30, color='skyblue', edgecolor='black')
plt.title('Log-Transformed Acousticness')
plt.xlabel('Log(Acousticness)')
plt.ylabel('Frequency')
plt.show()

df['acousticness_sqrt'] = np.sqrt(df['acousticness'])
plt.hist(df['acousticness_sqrt'], bins=30, color='orange', edgecolor='black')
plt.title('Square Root Transformed Acousticness')
plt.xlabel('Sqrt(Acousticness)')
plt.ylabel('Frequency')
plt.show()

#Preprocess


df = pd.get_dummies(df, columns=['key', 'mode'])

X = df.drop(columns=['music_genre'])  
y = df['music_genre']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state= rs)

print(y_train.value_counts())
print(y_test.value_counts())

numerical_cols = X_train.select_dtypes(include='number').columns

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


pca = PCA(n_components= 0.95)
X_train_pca = pca.fit_transform(X_train[numerical_cols])
X_test_pca = pca.transform(X_test[numerical_cols])


eigenvalues = pca.explained_variance_
print("Eigenvalues:", eigenvalues)
print(f"Number of components with eigenvalue > 1: {(eigenvalues > 1).sum()}")

explained_var = pca.explained_variance_ratio_.cumsum()
plt.plot(np.arange(1, len(explained_var) + 1), explained_var)
plt.axhline(0.95, color='red', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()


#Modeling

#Logistic

logRe = LogisticRegression(
    max_iter=1000,
    random_state=rs
)
logRe.fit(X_train_pca, y_train)
y_pred_log = logRe.predict(X_test_pca)

accuracy_logreg = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", accuracy_logreg)
y_proba = logRe.predict_proba(X_test_pca)
auc_score = roc_auc_score(y_test, y_proba, average='macro', multi_class='ovr')
print(auc_score)

#RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=rs)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for randomForest Model:", accuracy)
y_prob = clf.predict_proba(X_test_pca)

auc_score = roc_auc_score(y_test, y_prob,multi_class='ovr')
print(f"AUC score for randomForest Model: {auc_score}")

#SVM
svm = SVC(C=1.0, gamma='scale', kernel='rbf', probability=True)
svm.fit(X_train_pca,y_train)
y_pred_svc = svm.predict(X_test_pca)
acc_svc = accuracy_score(y_test, y_pred_svc)
print("SVM Accuracy:", acc_svc)
y_proba_svc = svm.predict_proba(X_test_pca)
auc_svc = roc_auc_score(y_test, y_proba_svc, average='macro', multi_class='ovr')
print("SVM AUC:", auc_svc)


#FNN with PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

genre_list = sorted(df['music_genre'].unique())
genre_to_int = {genre: idx for idx, genre in enumerate(genre_list)}

y_train_encoded = [genre_to_int[label] for label in y_train]
y_test_encoded = [genre_to_int[label] for label in y_test]

X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)



class FNN(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(FNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.network(x)
    
model = FNN(input_size=X_train_pca.shape[1], n_hidden=64, output_size=10).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
def train_model(model, X_train, y_train, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

train_model(model, X_train_tensor, y_train_tensor, epochs=100)
model.eval()
with torch.no_grad():
    output = model(X_test_tensor)
    probs = torch.exp(output).cpu().numpy()
    preds = np.argmax(probs, axis=1)

accuracy = accuracy_score(y_test_encoded, preds)
auc = roc_auc_score(y_test_encoded,probs, average='macro', multi_class='ovr')
print("FNN:", accuracy)
print("FNN:", auc)

        
    


























