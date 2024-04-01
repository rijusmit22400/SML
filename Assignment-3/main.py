import numpy as np

# Load the MNIST dataset
mnist_data = np.load('mnist.npz')

# Access the arrays within the npz file
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']
x_test = mnist_data['x_test']
y_test = mnist_data['y_test']

# Filter the dataset to select classes 0, 1, and 2
train_filter = np.isin(y_train, [0, 1, 2])
test_filter = np.isin(y_test, [0, 1, 2])

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

# Reshape the images to 784*18623
x_train_filtered_vector = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
x_test_filtered_vector = x_test_filtered.reshape(x_test_filtered.shape[0], -1)

# Apply PCA and reduce the dimension to p = 10
mean_train = np.mean(x_train_filtered_vector, axis=0)
centered_train = x_train_filtered_vector - mean_train
cov_train = np.dot(centered_train.T, centered_train) / (x_train_filtered_vector.shape[0] - 1)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(cov_train)

# Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_indices]
sorted_eigenvectors = eigen_vectors[:, sorted_indices]

# Create the U matrix for PCA using the first 10 eigenvectors
U = sorted_eigenvectors[:, :10]
Y_train = np.dot(centered_train, U)
Y_train = Y_train.astype(np.float32)

# Reconstruct the original data using the first 10 eigenvectors
x_train_filtered_vector_recon = Y_train
x_test_filtered_vector_recon = np.dot(x_test_filtered_vector - np.mean(x_test_filtered_vector), U)
x_test_filtered_vector_recon = x_test_filtered_vector_recon.astype(np.float32)

# Define function to calculate Gini index
def calculate_gini_index_optimized(feature, labels):
    """
    Calculate the Gini index for all possible split values of a feature and return the minimum.

    Parameters:
        feature (numpy.ndarray): Array containing the feature.
        labels (numpy.ndarray): Array containing the class labels.

    Returns:
        float: The minimum Gini index.
        float: The value in the feature which yielded the minimum Gini index.
    """
    # Sort the feature values and corresponding labels
    sorted_indices = np.argsort(feature)
    sorted_feature = feature[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Initialize variables
    n = len(feature)
    min_gini_index = float('inf')
    best_split_value = None

    # Initialize counts for the left and right subsets
    left_counts = {label: 0 for label in np.unique(labels)}
    right_counts = {label: np.sum(labels == label) for label in np.unique(labels)}
    
    # Initialize the left and right pointers
    left_ptr = 0
    right_ptr = n - 1

    # Binary search for the split value with minimum Gini index
    while left_ptr < right_ptr:
        mid = (left_ptr + right_ptr) // 2
        value = sorted_feature[mid]
        
        # Update counts for left and right subsets
        for i in range(left_ptr, mid + 1):
            left_counts[sorted_labels[i]] += 1
        for i in range(mid + 1, right_ptr + 1):
            right_counts[sorted_labels[i]] -= 1
        
        # Calculate Gini index for the current split
        left_total = mid - left_ptr + 1
        right_total = right_ptr - mid
        total_samples = n
        
        left_gini = 1 - sum((left_counts[label] / left_total)**2 for label in left_counts)
        right_gini = 1 - sum((right_counts[label] / right_total)**2 for label in right_counts)
        
        weighted_gini = (left_total / total_samples) * left_gini + (right_total / total_samples) * right_gini
        
        # Update minimum Gini index and best split value if necessary
        if weighted_gini < min_gini_index:
            min_gini_index = weighted_gini
            best_split_value = value
        
        # Move the pointers based on the Gini index of the left and right subsets
        if left_gini < right_gini:
            left_ptr = mid + 1
        else:
            right_ptr = mid

    return min_gini_index, best_split_value

# Define function to find best split for a given dimension
def find_best_split(data, labels):
    """
    Find the best split value for each feature using binary search.

    Parameters:
        data (numpy.ndarray): Array containing the features.
        labels (numpy.ndarray): Array containing the class labels.

    Returns:
        tuple: The best split value and the index where the best split value was found.
    """
    n_features = data.shape[1]
    best_split_values = np.zeros(n_features)
    gini_indices = np.zeros(n_features)

    # Calculate Gini index for each feature
    for i in range(n_features):
        gini_indices[i], best_split_values[i] = calculate_gini_index_optimized(data[:, i], labels)

    # Binary search to find the best split value
    left = 0
    right = n_features - 1
    best_gini_index = float('inf')
    best_split_value = None
    best_index = None

    while left <= right:
        mid = (left + right) // 2
        gini_index = gini_indices[mid]

        if gini_index < best_gini_index:
            best_gini_index = gini_index
            best_split_value = best_split_values[mid]
            best_index = mid

        if mid > 0 and gini_indices[mid - 1] < gini_indices[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return best_split_value, best_index

# Test the function
best_split_value, best_index = find_best_split(x_train_filtered_vector_recon, y_train_filtered)

# Define function to grow decision tree
def grow_decision_tree(data, labels, max_depth=3):
    """
    Grow a decision tree classifier.

    Parameters:
        data (numpy.ndarray): Array containing the features.
        labels (numpy.ndarray): Array containing the class labels.
        max_depth (int): Maximum depth of the decision tree.

    Returns:
        dict: The decision tree.
    """
    if max_depth == 0 or len(np.unique(labels)) == 1:
        # If maximum depth is reached or all labels are of the same class,
        # return the most frequent class
        return np.bincount(labels).argmax()

    # Find the best split value and index
    best_split_value, best_index = find_best_split(data, labels)

    if best_index is None:
        # If no further split decreases Gini index, return the most frequent class
        return np.bincount(labels).argmax()

    # Split the dataset based on the best split value
    left_indices = data[:, best_index] <= best_split_value
    right_indices = ~left_indices

    # Grow left and right subtrees
    left_subtree = grow_decision_tree(data[left_indices], labels[left_indices], max_depth - 1)
    right_subtree = grow_decision_tree(data[right_indices], labels[right_indices], max_depth - 1)

    # Return decision tree node
    return {'dimension': best_index, 'split_value': best_split_value,
            'left_subtree': left_subtree, 'right_subtree': right_subtree}

# Grow decision tree
decision_tree = grow_decision_tree(x_train_filtered_vector_recon, y_train_filtered)

# Function to classify test samples using decision tree
def classify_test_samples(tree, data):
    """
    Classify test samples using a decision tree.

    Parameters:
        tree (dict): The decision tree.
        data (numpy.ndarray): Array containing the test samples.

    Returns:
        numpy.ndarray: Predicted class labels.
    """
    predictions = np.zeros(len(data))
    for i, sample in enumerate(data):
        node = tree
        while isinstance(node, dict):
            dimension = node['dimension']
            split_value = node['split_value']
            if sample[dimension] <= split_value:
                node = node['left_subtree']
            else:
                node = node['right_subtree']
        predictions[i] = node
    return predictions

# Test the classifier
test_predictions = classify_test_samples(decision_tree, x_test_filtered_vector_recon)
unique_values = np.unique(test_predictions)
print("Unique values in test predictions:", unique_values)
# # Calculate accuracy and class-wise accuracy for the testing dataset
# accuracy = np.mean(test_predictions == y_test_filtered)
# class_wise_accuracy = {c: np.mean(test_predictions[y_test_filtered == c] == c) for c in np.unique(y_test_filtered)}

# print("Decision Tree Accuracy:", accuracy)
# print("Decision Tree Class-wise Accuracy:", class_wise_accuracy)

# # Now use bagging
# def bagging(x_train, y_train, n_datasets=5):
#     trees = []
#     for i in range(n_datasets):
#         indices = np.random.choice(len(x_train), len(x_train), replace=True)
#         x_subset = x_train[indices]
#         y_subset = y_train[indices]
#         tree = grow_decision_tree(x_subset, y_subset)
#         trees.append(tree)
#     return trees

# # Grow decision trees using bagging
# trees = bagging(x_train_filtered_vector_recon, y_train_filtered)

# # Function to predict class labels using majority voting
# def majority_voting(trees, data):
#     predictions = np.array([np.array([classify_test_samples(tree, sample) for sample in data]) for tree in trees])
#     majority_predictions = np.array([np.bincount(predictions[:, :, i]).argmax() for i in range(len(data))])
#     return majority_predictions

# # Classify test samples using bagging
# bagging_predictions = majority_voting(trees, x_test_filtered_vector_recon)

# # Calculate total accuracy and class-wise accuracy for bagging
# bagging_accuracy = np.mean(bagging_predictions == y_test_filtered)
# bagging_class_wise_accuracy = {c: np.mean(bagging_predictions[y_test_filtered == c] == c) for c in np.unique(y_test_filtered)}

# print("Bagging Accuracy:", bagging_accuracy)
# print("Bagging Class-wise Accuracy:", bagging_class_wise_accuracy)
