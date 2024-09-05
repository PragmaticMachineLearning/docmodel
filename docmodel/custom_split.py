import numpy as np

def custom_train_test_split(data, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    total_samples = len(data[0])
    test_samples = int(total_samples * test_size)
    indices = np.random.permutation(total_samples)
    train_idx, test_idx = indices[test_samples:], indices[:test_samples]

    train_data = [[item[i] for i in train_idx] for item in data]
    test_data = [[item[i] for i in test_idx] for item in data]

    return train_data, test_data
