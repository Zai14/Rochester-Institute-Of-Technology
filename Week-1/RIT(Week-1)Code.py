import pandas as pd
import numpy as np

# Load the CSV file
file_path = r"C:\Users\zaids\Downloads\Uncleaned(RIT).csv"
df = pd.read_csv(file_path)

# 1. Data Cleaning and Validation

# Convert date columns to datetime format
date_columns = ['Learner SignUp DateTime', 'Opportunity End Date', 'Date of Birth', 
                'Entry created at', 'Apply Date', 'Opportunity Start Date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Remove duplicates if any
df_cleaned = df.drop_duplicates()

# Check for missing values
missing_values = df_cleaned.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values
# Drop 'Institution Name' missing entries if needed (since there are only 4)
df_cleaned['Institution Name'].fillna('Unknown', inplace=True)

# For columns with larger missing values, decide to impute or remove them based on business logic
# Fill missing 'Opportunity End Date' and 'Opportunity Start Date' where necessary
df_cleaned['Opportunity End Date'].fillna(pd.Timestamp('today'), inplace=True)
df_cleaned['Opportunity Start Date'].fillna(df_cleaned['Opportunity End Date'], inplace=True)

# 2. Feature Engineering

# Feature 1: Calculate Age from Date of Birth
current_date = pd.Timestamp('today')
df_cleaned['Age'] = df_cleaned['Date of Birth'].apply(lambda dob: current_date.year - dob.year)

# Feature 2: Calculate Opportunity Duration (in days)
df_cleaned['Opportunity Duration (days)'] = (df_cleaned['Opportunity End Date'] - df_cleaned['Opportunity Start Date']).dt.days

# Fill missing 'Opportunity Duration' with a placeholder value, e.g., -1 for unknown durations
df_cleaned['Opportunity Duration (days)'].fillna(-1, inplace=True)

# Preview the new features
print(df_cleaned[['Age', 'Opportunity Duration (days)']].head())

# Save the cleaned and transformed data to a new CSV file (optional)
output_path = r"C:\Users\zaids\Downloads\Wd_cleaned.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"Data cleaned and saved to {output_path}")
