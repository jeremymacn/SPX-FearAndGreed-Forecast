import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def plot_roc_curve(y_test, y_pred_proba, output_file='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close()
    print(f"ROC curve saved to {output_file}")

def plot_confusion_matrix_heatmap(y_test, y_pred, output_file='confusion_matrix_heatmap.png'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_file)
    plt.close()
    print(f"Confusion matrix heatmap saved to {output_file}")


def plot_feature_importance(model, X, output_file='feature_importance.png'):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Feature importance plot saved to {output_file}")


def train_model(df, target_column='Target_21d'):
    """
    Trains a RandomForestClassifier model and evaluates it.
    """
    # Select features and target
    X = df.drop(columns=[col for col in df.columns if 'Target' in col])
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print(f"--- Model Evaluation for {target_column} ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nAUC Score:")
    print(roc_auc_score(y_test, y_pred_proba))

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)

    # Visualize the results
    print("\nGenerating model visualizations...")
    plot_roc_curve(y_test, y_pred_proba)
    plot_confusion_matrix_heatmap(y_test, y_pred)
    plot_feature_importance(model, X)
