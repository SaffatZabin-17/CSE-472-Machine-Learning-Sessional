import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from logistic_regression import _LogisticRegression

class _BaggingEnsemble:
    def __init__(self, n_estimators=9, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.bootstrapped_datasets = []
        self.all_predictions = None

    def create_bootstrap_samples(self, feature_data_train, target_data_train):
        
        for i in range(self.n_estimators):
            X_resampled, y_resampled = resample(feature_data_train, target_data_train, replace=True, 
                                                n_samples=len(feature_data_train), random_state=self.random_state+i)
            self.bootstrapped_datasets.append((X_resampled, y_resampled))

    def fit(self, feature_data_train, target_data_train):
        
        self.create_bootstrap_samples(feature_data_train, target_data_train)

        for i, (X_resampled, y_resampled) in enumerate(self.bootstrapped_datasets):
            model = _LogisticRegression()
            model.fit(X_resampled, y_resampled)
            self.models.append(model)

    def predict_all(self, X_test):
        
        n_samples_test = X_test.shape[0]
        self.all_predictions = np.zeros((n_samples_test, self.n_estimators))

        for i, model in enumerate(self.models):
            y_pred = model.predict(X_test)
            self.all_predictions[:, i] = y_pred

        return self.all_predictions

    def final_prediction(self, X_test=None):
        
        if X_test is not None:
            self.predict_all(X_test)

        n_samples_test = self.all_predictions.shape[0]
        final_prediction = np.zeros(n_samples_test)

        for i in range(n_samples_test):
            unique_labels, counts = np.unique(self.all_predictions[i, :], return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            final_prediction[i] = majority_label

        return final_prediction
