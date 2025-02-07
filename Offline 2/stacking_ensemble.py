import numpy as np
import pandas as pd
from bagging_ensemble import _BaggingEnsemble
from logistic_regression import _LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': specificity,
        'f1_score': f1_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_pred),
        'aupr': average_precision_score(y_true, y_pred)
    }
    
    return metrics

class _StackingEnsemble:
    def __init__(self, base_model=_LogisticRegression, meta_model=_LogisticRegression, n_estimators=9):

        self.base_model = base_model
        self.meta_model = meta_model
        self.n_estimators = n_estimators
        self.base_models = []
        self.meta_features_train = None
        self.metrics_df = None

    def fit(self, X_train, y_train, validation_size=0.2):
        
        X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

        self.base_models = []
        meta_features_val = np.zeros((X_val.shape[0], self.n_estimators))
        model_metrics = []

        for i in range(self.n_estimators):

            X_resampled, y_resampled = resample(X_train_main, y_train_main, random_state=i)

            model = _LogisticRegression()
            model.fit(X_resampled, y_resampled)
            self.base_models.append(model)

            y_pred_val = model.predict(X_val)
            meta_features_val[:, i] = y_pred_val

            metrics = calculate_metrics(y_val, y_pred_val)
            model_metrics.append(metrics)

        self.meta_features_train = np.concatenate([X_val, meta_features_val], axis=1)

        self.meta_model.fit(self.meta_features_train, y_val)

        self.metrics_df = pd.DataFrame(model_metrics)

    def plot_violin_plots(self):
        #for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auroc', 'aupr']:
        #    plt.figure(figsize=(10, 6))
        #    sns.violinplot(data=self.metrics_df[metric], inner="quartile")
        #    plt.title(f'Violin Plot for {metric.capitalize()}')
        #    plt.ylabel(metric.capitalize())
        #    plt.xlabel('Base Models')
        #    plt.show()

        # Melt the DataFrame for easier plotting
        melted_metrics_df = pd.melt(self.metrics_df, var_name='Metric', value_name='Score')

        # Set a color palette to distinguish the metrics
        palette = sns.color_palette("husl", len(self.metrics_df.columns))  # Color palette for different metrics

        # Plot all metrics in a single violin plot with colors
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Metric', y='Score', data=melted_metrics_df, inner="point", palette=palette)
        
        # Add title and labels
        plt.title('Violin Plot for All Metrics Across Base Models')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        
        # Adjust x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adding a caption as seen in your image
        plt.figtext(0.5, -0.05, "Fig: Violin Plot", ha="center", fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def predict(self, X_test):
        meta_features_test = np.zeros((X_test.shape[0], self.n_estimators))

        for i, model in enumerate(self.base_models):
            y_pred_test = model.predict(X_test)
            meta_features_test[:, i] = y_pred_test

        combined_test_features = np.concatenate([X_test, meta_features_test], axis=1)

        final_predictions = self.meta_model.predict(combined_test_features)

        return final_predictions

