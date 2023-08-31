import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Add the names of columns
column_labels = ['Subject number', 'Jitter in %', 'Abs Jitter', 'Jitter as RAP', 'Jitter as PPQ5', 'Jitter as DDP', 'Shimmer in %', 'Abs Shimmer', 'Shimme as APQ3', 'Shimmer as APQ5', 'Shimmer as APQ11', 'Shimmer as DDA', 'Harmonicity (Autocorrelation)', 'Noise to Hermonic ratio', 'Hermonic to Noise Ratio', 'Median Pitch', 'Mean Pitch', 'Standard deviation of Pitch', 'Minimum pitch', 'Maximum pitch', 'Number of pulses', 'Number of periods', 'Mean period', 'Standard deviation of period', 'Fraction of locally unvoiced frames', 'Number of voice breaks', 'Degree of voice breaks', 'UPDRS', 'PD indicator']

df1 = pd.read_csv('po1_data.csv', names=column_labels)

# Save new CSV file
df1.to_csv('po1_newdata.csv', index=False)

# Load new file
df1 = pd.read_csv('po1_newdata.csv')

# Filter rows for individuals with Parkinson's disease (PD indicator = 1)
df1_pd = df1[df1['PD indicator'] == 1].iloc[:, 1:-2]

# Filter rows for individuals without Parkinson's disease (PD indicator = 0)
df1_non_pd = df1[df1['PD indicator'] == 1].iloc[:, 1:-2]

# Calculate descriptive statistics (mean and standard deviation) for PD group
pd_stats = df1_pd.describe().loc[['mean', 'std']]

# Calculate descriptive statistics (mean and standard deviation) for non-PD group
non_pd_stats = df1_non_pd.describe().loc[['mean', 'std']]

# Print the calculated statistics for the PD group
print("Descriptive Statistics for Individuals with Parkinson's Disease:")
print(pd_stats)

# Print the calculated statistics for the non-PD group
print("\nDescriptive Statistics for Individuals without Parkinson's Disease:")
print(non_pd_stats)

# Save the descriptive statistics to a CSV file
stats_combined = pd.concat([pd_stats, non_pd_stats], keys=['PD Group', 'Non-PD Group'])
stats_combined.to_csv('descriptive_stats_mean_std.csv')

# Separate the dataframe into two groups: those with PD and those without
ppd = df1[df1['PD indicator'] == 1]
non_ppd = df1[df1['PD indicator'] == 0]

# Perform a t-test for each feature
salient_variables = []
for i in column_labels:
     t_stat, p_val = ttest_ind(ppd[i], non_ppd[i])
     if p_val < 0.05:
         print(f'Feature: {i}, T-statistic: {t_stat}, P-value: {p_val}')
