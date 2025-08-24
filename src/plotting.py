import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import logging
import os

def plot_roc_curve(y_test, y_pred_proba, output_dir='reports/images', filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    output_path = os.path.join(output_dir, filename)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"ROC curve saved to {output_path}")

def plot_confusion_matrix_heatmap(y_test, y_pred, output_dir='reports/images', filename='confusion_matrix_heatmap.png'):
    cm = confusion_matrix(y_test, y_pred)
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix heatmap saved to {output_path}")


def plot_feature_importance(model, X, output_dir='reports/images', filename='feature_importance.png'):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(12, 10))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Feature importance plot saved to {output_path}")

def plot_correlation_matrix(df, output_dir='reports', filename='correlation_matrix.png'):
    """
    Plots the correlation matrix of the dataframe.
    """
    output_path = os.path.join(output_dir, filename)
    plt.figure(figsize=(18, 15))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Correlation matrix saved to {output_path}")

def plot_target_distribution(df, output_dir='reports/images', filename='target_distribution.png'):
    """
    Plots the distribution of the target variable.
    """
    output_path = os.path.join(output_dir, filename)
    target_cols = [col for col in df.columns if 'Target' in col]

    fig, axes = plt.subplots(len(target_cols), 1, figsize=(6, 4 * len(target_cols)))
    if len(target_cols) > 1:
        fig.suptitle('Target Variable Distributions')

    for i, target in enumerate(target_cols):
        ax = axes[i] if len(target_cols) > 1 else axes
        sns.countplot(ax=ax, x=target, data=df)
        ax.set_title(f'Distribution of {target}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Target distribution plot saved to {output_path}")
