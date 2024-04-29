import numpy as np

# Load the MNIST dataset
mnist_data = np.load(r'C:\Users\rijus\OneDrive\Desktop\Projects\SML\Assignment-4\mnist.npz')

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

# Apply PCA and reduce the dimension to p = 5
mean_train = np.mean(x_train_filtered_vector, axis=0)
centered_train = x_train_filtered_vector - mean_train
cov_train = np.dot(centered_train.T, centered_train) / (x_train_filtered_vector.shape[0] - 1)

eigen_values, eigen_vectors = np.linalg.eig(cov_train)
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_indices]
sorted_eigenvectors = eigen_vectors[:, sorted_indices]

U = sorted_eigenvectors[:, :5]
x_train_filtered_vector_recon = np.dot(centered_train, U).astype(np.float32)
x_test_filtered_vector_recon = np.dot(x_test_filtered_vector - np.mean(x_test_filtered_vector), U).astype(np.float32)
x_val = np.dot(x_val - np.mean(x_val), U).astype(np.float32)

class DecisionStump:
    def __init__(self):
        self.split_dim = None
        self.split_value = None
        self.min_ssr = float('inf')
        
    def train(self, X, y):
        n_samples, n_features = X.shape
        
        # Iterate over each dimension
        for dim in range(n_features):
            unique_values = np.unique(X[:, dim])
            unique_values.sort()
            
            # Calculate midpoints between consecutive unique values
            split_points = (unique_values[:-1] + unique_values[1:]) / 2
            
            # Evaluate each split point
            for split_value in split_points:
                # Split the data
                left_indices = X[:, dim] <= split_value
                right_indices = ~left_indices
                
                # Calculate SSR for left and right partitions
                left_mean = np.mean(y[left_indices])
                right_mean = np.mean(y[right_indices])
                left_ssr = np.sum((y[left_indices] - left_mean) ** 2)
                right_ssr = np.sum((y[right_indices] - right_mean) ** 2)
                total_ssr = left_ssr + right_ssr
                
                # Update the best split if necessary
                if total_ssr < self.min_ssr:
                    self.min_ssr = total_ssr
                    self.split_dim = dim
                    self.split_value = split_value
    
    def predict(self, X):
        return X[:, self.split_dim] <= self.split_value

# Learn h1(x)
decision_stump = DecisionStump()
decision_stump.train(x_train_filtered_vector_recon, y_train_filtered)

print("Best split dimension:", decision_stump.split_dim)
print("Best split value:", decision_stump.split_value)

# Compute residue using y - 0.01*h1(x)
y_train_residual = y_train_filtered - 0.01 * decision_stump.predict(x_train_filtered_vector_recon)

# Learn h2(x) using the train set with updated labels
h2_decision_stump = DecisionStump()
h2_decision_stump.train(x_train_filtered_vector_recon, y_train_residual)

print("Best split dimension for h2(x):", h2_decision_stump.split_dim)
print("Best split value for h2(x):", h2_decision_stump.split_value)

# Compute residue using y - 0.01*h1(x) - 0.01*h2(x)
y_train_residual = y_train_filtered - 0.01 * decision_stump.predict(x_train_filtered_vector_recon) - 0.01 * decision_stump.predict(x_train_filtered_vector_recon)

