from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Initialize the RandomForest model
model = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, data_resampled, labels_resampled, cv=5)
print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')
