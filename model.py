import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
val_images = np.load('val_images.npy')
val_labels = np.load('val_labels.npy')

# SVM model
def create_svm_model(train_images, train_labels):
    svm_model = SVC(kernel='linear')
    svm_model.fit(train_images.reshape(train_images.shape[0], -1), train_labels)
    return svm_model

# Random Forest model
def create_random_forest_model(train_images, train_labels):
    rf_model = RandomForestClassifier()
    rf_model.fit(train_images.reshape(train_images.shape[0], -1), train_labels)
    return rf_model

# Function to evaluate models
def evaluate_models(models, test_images, test_labels):
    predictions = np.zeros((len(models), len(test_labels)))
    for i, model in enumerate(models):
        predictions[i, :] = model.predict(test_images.reshape(test_images.shape[0], -1))
    majority_vote = np.round(np.mean(predictions, axis=0)).astype(int)
    accuracy = accuracy_score(test_labels, majority_vote)
    return accuracy

if __name__ == '__main__':
    # Create SVM model
    svm_model = create_svm_model(train_images, train_labels)
    svm_accuracy = svm_model.score(val_images.reshape(val_images.shape[0], -1), val_labels)
    print(f"SVM Accuracy: {svm_accuracy}")

    # Create Random Forest model
    rf_model = create_random_forest_model(train_images, train_labels)
    rf_accuracy = rf_model.score(val_images.reshape(val_images.shape[0], -1), val_labels)
    print(f"Random Forest Accuracy: {rf_accuracy}")

    # Evaluate ensemble model
    models = [svm_model, rf_model]
    ensemble_accuracy = evaluate_models(models, val_images, val_labels)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy}")

    # Save models
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Models saved as svm_model.pkl and rf_model.pkl")
