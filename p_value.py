import pandas as pd
from scipy.stats import pearsonr


df = pd.read_csv('data\\p_value_agro.csv', delimiter=';', skipinitialspace=True)
col1, col2 = df.columns
df = df.dropna()


corr, p_value = pearsonr(df[col1], df[col2])


print(f"Корреляция между '{col1}' и '{col2}': {corr:.4f}")
print(f"P-value: {p_value:.6f}")
