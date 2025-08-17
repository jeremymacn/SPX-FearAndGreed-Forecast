import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

def plot_correlation_matrix(df, output_file='correlation_matrix.png'):
    """
    Plots the correlation matrix of the dataframe.
    """
    plt.figure(figsize=(18, 15))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(output_file)
    plt.close()
    print(f"Correlation matrix saved to {output_file}")

def plot_target_distribution(df, output_file='target_distribution.png'):
    """
    Plots the distribution of the target variable.
    """
    target_cols = [col for col in df.columns if 'Target' in col]
    
    fig, axes = plt.subplots(len(target_cols), 1, figsize=(6, 4 * len(target_cols)))
    fig.suptitle('Target Variable Distributions')

    for i, target in enumerate(target_cols):
        sns.countplot(ax=axes[i], x=target, data=df)
        axes[i].set_title(f'Distribution of {target}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"Target distribution plot saved to {output_file}")

def calculate_mutual_information(df):
    """
    Calculates and prints the mutual information scores for each feature.
    """
    target_cols = [col for col in df.columns if 'Target' in col]
    X = df.drop(columns=target_cols)

    for target in target_cols:
        print(f"\n--- Mutual Information for {target} ---")
        y = df[target]
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        print(mi_series)

def run_eda(df):
    """
    Runs the exploratory data analysis.
    """
    print("\nPlotting correlation matrix...")
    plot_correlation_matrix(df)
    print("\nPlotting target distribution...")
    plot_target_distribution(df)
    print("\nCalculating Mutual Information...")
    calculate_mutual_information(df)
