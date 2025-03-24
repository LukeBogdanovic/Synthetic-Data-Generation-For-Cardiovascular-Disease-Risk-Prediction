'''
:File: data_augmentation_script.py
:Author: Luke Bogdanovic
:Date: 12/03/2025
:Purpose: Oversamples the CRFs in the dataset before being used for generating new samples with TVAE/CTGAN.
'''
# %%
from imblearn.over_sampling import SMOTEN, ADASYN
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
file_path = "../data/CRFs.csv"
df = pd.read_csv(file_path)  # Read the CSV file
# Replace n/a values with numpy nan values and drop all rows with nan values
df = df.replace("n/a", np.nan).dropna()

# %%
# Set text in gender column to uppercase
df['Gender'] = df['Gender'].str.upper()

# %%
continuous_cols = ['Age', 'Weight', 'Height',
                   'SBP', 'DBP']  # Continuous data columns
categorical_cols = ['Smoker', 'Gender']  # Categorical data columns
target_col = 'Vascular event'  # Class column

# %%
X = df.drop(columns=[target_col])  # Get dataset without the class column
y = df[target_col]  # Get class column

# %%
scaler = StandardScaler()  # Instantiate the standard scaler
# Fit and transform the continuous data
X_cont_scaled = scaler.fit_transform(X[continuous_cols])
# Create dataframe using the continous data
X_cont_scaled = pd.DataFrame(X_cont_scaled, columns=continuous_cols)

# %%
encoder = OrdinalEncoder()  # Instantiate the ordinal encoder
# Fit and transform the categorical data
X_cat_encoded = encoder.fit_transform(X[categorical_cols])
# Create dataframe using the categorical data
X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=categorical_cols)

# %%
X_cont_scaled.head()  # Get first 5 continuous scaled rows - used for jupyter notebook only

# %%
# Get first 5 categorical encoded rows - used for jupyter notebook only
X_cat_encoded.head()

# %%
print("Original class distribution:")
# Print the number of values in each class in original distribution
print(pd.Series(y).value_counts())

# %%
adasyn = ADASYN(sampling_strategy='not majority',
                n_neighbors=2, random_state=42)  # Instantiate the ADASYN oversampling method for continuous values
X_cont_res, y_cont_res = adasyn.fit_resample(
    X_cont_scaled, y)  # Fit and resample the data
# Print the number of values in each class in original distribution
vals = pd.Series(y_cont_res).value_counts()
# Number of samples to oversample using SMOTEN
smote_oversample = {
    'myocardial infarction': vals.values[0],
    'stroke': vals.stroke,
    'syncope': vals.syncope,
    'none': vals.none
}
print("After ADASYN (continuous features):")
# Print class values for continuous values
print(pd.Series(y_cont_res).value_counts())

# %%
smoten = SMOTEN(sampling_strategy=smote_oversample,  # Instantiate the SMOTEN oversampling method for categorical values
                k_neighbors=2, random_state=42)
X_cat_res, y_cat_res = smoten.fit_resample(
    X_cat_encoded, y)  # Fit and resample the data

print("After SMOTEN (categorical features):")
# Print class values for categorical values
print(pd.Series(y_cat_res).value_counts())

# %%
X_res = pd.concat([X_cont_res.reset_index(drop=True),
                   X_cat_res.reset_index(drop=True)], axis=1)  # Concatenate the continuous and categorical data together
y_res = y_cont_res.reset_index(drop=True)  # Reset index for the labels

# %%
X_res.head()  # Print first 5 rows of the oversampled data

# %%
y_res.head()  # Print first 5 rows of the oversampled labels

# %%
# Concatenate the data and labels together
augmented_data = pd.concat([X_res, y_res], axis=1)
augmented_data.head()  # Print first 5 rows of the augmented CRF dataset

# %%
augmented_data['Vascular event'].value_counts().plot(
    kind='bar', title='Class Distribution')  # Plot the classes in oversampled data
plt.show()  # Show the plot of classes in oversampled data

# %%
# Save the augmented data to a CSV
augmented_data.to_csv("augmented_dataset.csv", index=False)
