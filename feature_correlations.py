import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Case_Information.csv", encoding="ISO-8859-1")

# Print data types of columns
print("Before cleaning:")
print(df.dtypes)

# Clean up the dataset:
# Remove the 'ï»¿case_id' column (which seems to have encoding issues)
df = df.drop(columns=['ï»¿case_id'], errors='ignore')

# Convert 'age' to numeric, forcing errors to NaN
df["age"] = pd.to_numeric(df["age"], errors='coerce')

# Convert 'date_announced' and 'date_of_onset_of_symptoms' to datetime, and then to numeric (days since reference date)
df["date_announced"] = pd.to_datetime(df["date_announced"], errors='coerce')
df["date_of_onset_of_symptoms"] = pd.to_datetime(df["date_of_onset_of_symptoms"], errors='coerce')

# Convert dates to the number of days since a reference date (e.g., '2020-01-01')
reference_date = pd.to_datetime("2020-01-01")
df["date_announced"] = (df["date_announced"] - reference_date).dt.days
df["date_of_onset_of_symptoms"] = (df["date_of_onset_of_symptoms"] - reference_date).dt.days

# Encode categorical columns using LabelEncoder (for 'sex', 'home_quarantined', 'pregnant')
label_encoders = {}
for col in ["sex", "home_quarantined", "pregnant"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Drop rows with missing critical columns
df = df.dropna(subset=["age", "sex", "date_announced", "home_quarantined", "date_of_onset_of_symptoms", "pregnant"])

# Drop columns with all NaN values (optional: print them)
nan_columns = df.columns[df.isna().all()]
if len(nan_columns) > 0:
    print("Columns with all NaN values:", nan_columns)

# Drop columns with all NaN values
df = df.dropna(axis=1, how='all')

# After cleaning, print data types again
print("After cleaning:")
print(df.dtypes)

# Now, select only numeric columns for correlation computation
numeric_df = df.select_dtypes(include=[float, int])

# Verify if there are still numeric columns to perform correlation on
if numeric_df.empty:
    raise ValueError("Dataframe does not have enough numeric columns to compute correlation matrix.")

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()
