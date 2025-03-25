'''
:File: TVAE_script.py
:Author: Luke Bogdanovic
:Date: 12/03/2025
:Purpose: Trains the TVAE to create new synthetic CRFs using the oversampled CRFs.
'''
# %%
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE


# %%
df = pd.read_csv("../augmented_dataset.csv")  # Load the oversampled CSV file
df.head()

# %%
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)  # Detect metadata from the dataframe
model = TVAESynthesizer(metadata=metadata, epochs=3000,
                        cuda=False, verbose=True)  # Create TVAE model
model.fit(df)  # Fit the data
model.save("TVAE_model_cpu.pkl")  # Save the model
synthetic_crfs = model.sample(421)  # Sample data points
synthetic_crfs.to_csv("synthetic_crfs.csv", index=False)  # Save CRFs to file


# %%
# Pre-process data for ML models
target_column = "Vascular event"
X_real = df.drop(columns=[target_column])
X_real = X_real.fillna(X_real.mean())
y_real = df[target_column]
X_syn = synthetic_crfs.drop(columns=[target_column])
X_syn = X_syn.fillna(X_syn.mean())
y_syn = synthetic_crfs[target_column]

# %%
# Train RF classifier on real data and predict on synthetic
X_train, X_test, y_train, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred_syn = clf.predict(X_syn)
real_to_syn_acc = accuracy_score(y_syn, pred_syn)

# %%
# Train RF classifier on synthetic data and predict on real
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_syn, y_syn, test_size=0.2, random_state=42)
clf_syn = RandomForestClassifier()
clf_syn.fit(X_train_syn, y_train_syn)
pred_real = clf_syn.predict(X_real)
syn_to_real_acc = accuracy_score(y_real, pred_real)

print(f"Train on Real → Test on Synthetic Accuracy: {real_to_syn_acc:.4f}")
print(f"Train on Synthetic → Test on Real Accuracy: {syn_to_real_acc:.4f}")

# %%
print(
    f"Train on Real → Test on Synthetic Classification Report:\n{classification_report(y_syn, pred_syn)}")
print(
    f"Train on Synthetic → Test on Real Classification Report:\n{classification_report(y_real, pred_real)}")

# %%
# Train SVM on real data and predict on synthetic
svm_real = SVC()
svm_real.fit(X_train, y_train)
svm_real_pred = svm_real.predict(X_syn)
svm_real_to_syn_acc = accuracy_score(y_syn, svm_real_pred)
# Train SVM on synthetic data and predict on real
svm_syn = SVC()
svm_syn.fit(X_train_syn, y_train_syn)
svm_syn_pred = svm_syn.predict(X_real)
svm_syn_to_real_acc = accuracy_score(y_real, svm_syn_pred)

print(f"Train on Real → Test on Synthetic Accuracy: {svm_real_to_syn_acc:.4f}")
print(f"Train on Synthetic → Test on Real Accuracy: {svm_syn_to_real_acc:.4f}")

# %%
print(
    f"Train on Real → Test on Synthetic Classification Report:\n{classification_report(y_syn, pred_syn)}")
print(
    f"Train on Synthetic → Test on Real Classification Report:\n{classification_report(y_real, pred_real)}")

# %%
# Train KNN on real data and predict on synthetic
knn_real = KNeighborsClassifier()
knn_real.fit(X_train, y_train)
knn_real_pred = knn_real.predict(X_syn)
knn_real_to_syn_acc = accuracy_score(y_syn, knn_real_pred)
# Train KNN on synthetic data and predict on real
knn_syn = KNeighborsClassifier()
knn_syn.fit(X_train_syn, y_train_syn)
knn_syn_pred = knn_syn.predict(X_real)
knn_syn_to_real_acc = accuracy_score(y_real, knn_syn_pred)

print(
    f"Train on Real → Test on Synthetic (KNN) Accuracy: {knn_real_to_syn_acc:.4f}")
print(
    f"Train on Synthetic → Test on Real (KNN) Accuracy: {knn_syn_to_real_acc:.4f}")

# %%
print(
    f"Train on Real → Test on Synthetic Classification Report:\n{classification_report(y_syn, pred_syn)}")
print(
    f"Train on Synthetic → Test on Real Classification Report:\n{classification_report(y_real, pred_real)}")

# %%
# Create PCA visualization for the real and synthetic data points
pca = PCA(n_components=2)
real_pca = pca.fit_transform(X_real)
synthetic_pca = pca.transform(X_syn)
plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], label="Real Data", alpha=0.5)
plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1],
            label="Synthetic Data", alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.title("PCA Projection of Real vs. Synthetic Data")
plt.show()


# %%
# Create t-SNE visualization for the real and synthetic data points
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_real_tsne = tsne.fit_transform(X_real)
X_smote_tsne = tsne.fit_transform(X_syn)
plt.figure(figsize=(8, 6))
plt.scatter(X_real_tsne[:, 0], X_real_tsne[:, 1], label="Real Data", alpha=0.5)
plt.scatter(X_smote_tsne[:, 0], X_smote_tsne[:, 1],
            label="Synthetic Data", alpha=0.5)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.title("t-SNE Projection of Real vs. Synthetic Data")
plt.show()
