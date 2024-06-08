import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def pad_sequences(data, max_len):
    """Pad sequences to the same length."""
    padded_data = np.zeros((len(data), max_len))
    for i, seq in enumerate(data):
        if len(seq) > max_len:
            padded_data[i, :max_len] = seq[:max_len]
        else:
            padded_data[i, :len(seq)] = seq
    return padded_data

# Load data
try:
    with open('./datasibi.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    data = data_dict['data']
    labels = data_dict['labels']
except Exception as e:
    print("Error loading data: ", e)
    data = []
    labels = []

# Check if data and labels are loaded correctly
if not data or not labels:
    print("Data or labels are empty. Please check the contents of the pickle file.")
else:
    # Check the length of each entry
    lengths = [len(entry) for entry in data]
    max_len = max(lengths)
    print(f'Max length of sequences: {max_len}')

    # Pad sequences to the same length
    data_padded = pad_sequences(data, max_len)

    # Convert to numpy arrays
    data = np.array(data_padded)
    labels = np.array(labels)

    # Apply SMOTE to oversample the minority class
    smote = SMOTE()
    try:
        data_resampled, labels_resampled = smote.fit_resample(data, labels)
        # Check the shape after resampling
        print('Resampled data shape:', data_resampled.shape)
        print('Resampled labels shape:', labels_resampled.shape)

        # Perform stratified split using the resampled labels
        x_train, x_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, shuffle=True, stratify=labels_resampled)

        # Initialize and train the model
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # Make predictions
        y_predict = model.predict(x_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_predict)
        print('{}% of samples were classified correctly!'.format(accuracy * 100))

        # Save the model
        with open('modelsibi.p', 'wb') as f:
            pickle.dump({'model': model}, f)
    except Exception as e:
        print("Error during resampling or splitting: ", e)
