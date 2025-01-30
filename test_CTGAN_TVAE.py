from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata.metadata import Metadata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load real dataset
df = pd.read_csv("data/CRFs.csv")

df = df.drop(columns=['BMI', 'BSA', 'IMT MAX', 'LVMi', 'EF', 'Record'])

df['Gender'] = df['Gender'].str.upper().map({'M': 0, 'F': 1})
df['Smoker'] = df['Smoker'].str.upper().map({'NO': 0, 'YES': 1})

metadata = Metadata()
metadata.detect_from_dataframe(df)

# Choose Model: CTGAN or TVAE
use_ctgan = True  # Set to False to use TVAE instead

if use_ctgan:
    model = CTGANSynthesizer(metadata=metadata, epochs=500)
else:
    model = TVAESynthesizer(metadata=metadata, epochs=500)

# Train the model
model.fit(df)

# Generate synthetic data
synthetic_crfs = model.sample(1000)
synthetic_crfs.to_csv("synthetic_crfs.csv", index=False)

# Evaluation: Distribution Comparison
for column in df.columns:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[column], label="Real", fill=True, alpha=0.5)
    sns.kdeplot(synthetic_crfs[column],
                label="Synthetic", fill=True, alpha=0.5)
    plt.title(f"Comparison of Real vs. Synthetic for {column}")
    plt.legend()
    plt.show()

# Evaluation: Correlation Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(df.corr(), ax=axes[0], cmap="coolwarm", annot=True)
axes[0].set_title("Real Data Correlation")
sns.heatmap(synthetic_crfs.corr(), ax=axes[1], cmap="coolwarm", annot=True)
axes[1].set_title("Synthetic Data Correlation")
plt.show()

# Evaluation: Statistical Distance Metrics
for column in df.columns:
    wd = wasserstein_distance(df[column], synthetic_crfs[column])
    ks_stat, ks_p = ks_2samp(df[column], synthetic_crfs[column])
    print(f"Feature: {column}")
    print(f"  Wasserstein Distance: {wd:.4f}")
    print(f"  KS Test p-value: {ks_p:.4f}")
    print("-" * 30)

# Evaluation: ML Utility Test
# Assuming last column is the target (modify as needed)
target_column = "disease_outcome"
X_real = df.drop(columns=[target_column])
y_real = df[target_column]
X_syn = synthetic_crfs.drop(columns=[target_column])
y_syn = synthetic_crfs[target_column]

# Train on real, test on synthetic
X_train, X_test, y_train, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred_syn = clf.predict(X_syn)
real_to_syn_acc = accuracy_score(y_syn, pred_syn)

# Train on synthetic, test on real
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_syn, y_syn, test_size=0.2, random_state=42)
clf_syn = RandomForestClassifier()
clf_syn.fit(X_train_syn, y_train_syn)
pred_real = clf_syn.predict(X_real)
syn_to_real_acc = accuracy_score(y_real, pred_real)

print(f"Train on Real → Test on Synthetic Accuracy: {real_to_syn_acc:.4f}")
print(f"Train on Synthetic → Test on Real Accuracy: {syn_to_real_acc:.4f}")

# PCA Visualization
pca = PCA(n_components=2)
real_pca = pca.fit_transform(X_real)
synthetic_pca = pca.transform(X_syn)
plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], label="Real Data", alpha=0.5)
plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1],
            label="Synthetic Data", alpha=0.5)
plt.legend()
plt.title("PCA Projection of Real vs. Synthetic Data")
plt.show()
