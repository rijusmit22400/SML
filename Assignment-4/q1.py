import numpy as np
import matplotlib.pyplot as plt
import tree

# Load the MNIST dataset
mnist_data = np.load('mnist.npz')

# Access the arrays within the npz file
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']
x_test = mnist_data['x_test']
y_test = mnist_data['y_test']

# Filter the dataset to select classes 0, 1, and 2
train_filter = np.isin(y_train, [0, 1])
test_filter = np.isin(y_test, [0, 1])

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

# Reshape the images to 784*18623
x_train_filtered_vector = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
x_test_filtered_vector = x_test_filtered.reshape(x_test_filtered.shape[0], -1)

# Making the val set
val_size = 1000

# Get indices of class 0 and class 1 samples
class_0_indices = np.where(y_train_filtered == 0)[0]
class_1_indices = np.where(y_train_filtered == 1)[0]

# Shuffle the indices to ensure random selection
np.random.shuffle(class_0_indices)
np.random.shuffle(class_1_indices)
# Create the validation set and validation labels
x_val = np.concatenate((x_train_filtered_vector[class_0_indices[:val_size]], x_train_filtered_vector[class_1_indices[:val_size]]))
y_val = np.concatenate((y_train_filtered[class_0_indices[:val_size]], y_train_filtered[class_1_indices[:val_size]]))

# Remove the validation set samples from the train set and labels
x_train_filtered_vector = np.delete(x_train_filtered_vector, np.concatenate((class_0_indices[:val_size], class_1_indices[:val_size])), axis=0)
y_train_filtered = np.delete(y_train_filtered, np.concatenate((class_0_indices[:val_size], class_1_indices[:val_size])))

# Print the shapes of the train, validation, and test sets
print("Train set shape:", x_train_filtered_vector.shape)
print("Validation set shape:", x_val.shape)
print("Test set shape:", x_test_filtered_vector.shape)

#Train-set x_train_filtered_vector, y_train_filtered
#Val-set x_val, y_val
#Test-set x_test_filtered_vector, y_test_filtered

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

U = sorted_eigenvectors[:, :5]
Y_train = np.dot(centered_train, U)
Y_train = Y_train.astype(np.float32)

# Reconstruct the original data using the first 5 eigenvectors
x_train_filtered_vector_recon = Y_train
x_test_filtered_vector_recon = np.dot(x_test_filtered_vector - np.mean(x_test_filtered_vector), U)
x_test_filtered_vector_recon = x_test_filtered_vector_recon.astype(np.float32)
x_val = x_val @ U

print(x_test_filtered_vector_recon.shape)
print(x_train_filtered_vector_recon.shape)
print(x_val.shape)

def train_decision_stumps(X_train, y_train, X_val, y_val, n_stumps=300):
    stumps = []
    accuracies = []

    n_samples= X_train.shape[0]
    weights = np.ones(n_samples) / n_samples

    for i in range(n_stumps):
        stump = tree.DecisionStump(X_train, y_train, weights)
        stump.train()
        stumps.append(stump)

        predictions = stump.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        accuracies.append(accuracy)

        # Update weights for next iteration
        misclassified = predictions != y_val
        error_rate = np.sum(weights[misclassified]) / np.sum(weights)
        alpha = 0.5 * np.log((1 - error_rate) / error_rate)
        weights = np.where(misclassified, weights * np.exp(alpha), weights * np.exp(-alpha))
        weights /= np.sum(weights)

    return np.array(accuracies)

accuracies = train_decision_stumps(x_train_filtered_vector_recon, y_train_filtered, x_val, y_val)

def plot_array(array):
    x_values = range(1, len(array) + 1)
    plt.plot(x_values, array)
    plt.xlabel('Index + 1')
    plt.ylabel('Value')
    plt.title('Plot of Array Values')
    plt.grid(True)
    plt.show()
arr = [1, 2, 3, 4, 5]
plot_array(arr)
