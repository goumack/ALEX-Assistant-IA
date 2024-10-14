# visualization.py

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    numeric_columns = df.select_dtypes(include=['number'])
    plt.figure(figsize=(15, 9))
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='Wistia', fmt=".2f")
    plt.title('Correlation entre les differentes cultures')
    plt.show()
