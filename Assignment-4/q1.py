import numpy as np
import matplotlib.pyplot as plt

class DecisionStump:
    def __init__(self, train_set, label_data, weights=None):
        self.train_set = train_set
        self.label_data = label_data
        if weights is None:
            self.weights = np.ones(len(label_data)) / len(label_data)
        else:
            self.weights = weights
        self.split_value = None
        self.alpha = None
        self.index = None
        
    def find_best_split(self, feature, thresholds, total_sum_weights):
    # Reshape the feature vector to ensure proper broadcasting
        feature = feature.reshape(-1, 1)
        
        # Ensure thresholds is a numpy array
        thresholds = np.array(thresholds)
        
        # Broadcast feature and thresholds to enable element-wise comparison
        predictions = np.where(feature <= thresholds, 0, 1)
        
        # Initialize an array to store the L values
        L_values = np.zeros_like(thresholds, dtype=float)
        
        # Loop over each threshold to calculate L
        for i, threshold in enumerate(thresholds):
            # Calculate predictions for the current threshold
            errors = predictions[:, i] != self.label_data
            
            # Get the indices where there are no errors
            indices = np.where(~errors)
            
            # Calculate the sum of weights for this threshold
            sum_weights = np.sum(self.weights[indices])
            
            # Calculate L for this threshold
            L = sum_weights / total_sum_weights
            
            # Store the L value
            L_values[i] = L
        
        return L_values

    def classify(self, feature, threshold):
        return np.where(feature <= threshold, 0, 1)
    
    def train(self):
        best_splits = []
        """
        Train the decision stump using the weighted data
        """
        for i in range(self.train_set.shape[1]):
            feature = self.train_set[:, i]
            thresholds = np.unique(feature)
            thresholds.sort()
            new_thresholds = np.array([(thresholds[i] + thresholds[i+1]) / 2 for i in range(len(thresholds)-1)])
            new_thresholds = np.random.choice(thresholds, size=1000, replace=False)
            total_sum_weights = np.sum(self.weights)
            losses = self.find_best_split(feature, new_thresholds, total_sum_weights)
            best_split = thresholds[np.argmin(losses)]
            best_splits.append([best_split,np.min(losses)])
        best_splits=np.array(best_splits)
        j_values = best_splits[:, 1]
        min_index = np.argmin(j_values)
        self.split_value,L = best_splits[min_index]
        self.index = min_index
        #calculate alpha
        self.alpha = 0.5 * np.log((1 - L) / L)
        #new classification for updating the weights
        predictions = np.where(self.train_set[:, min_index] <= self.split_value, 0, 1)
        #update the weights
        missclassifed_indices = np.where(predictions != self.label_data)
        false_indices = missclassifed_indices[missclassifed_indices == False]
        #update the weights
        self.weights[false_indices] = self.weights[false_indices] * np.exp(self.alpha)

    def min_loss(self, feature, threshold):
        predictions = np.where(feature <= threshold, 0, 1)
        loss = np.sum(self.weights * (predictions != self.label_data))
        return loss

    def get_weights(self):
        return self.weights
    
    def predict(self, X):
        feature = X[:, self.index]
        return np.where(feature <= self.split_value, 0, 1)

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
x_val = np.dot(x_val - np.mean(x_val), U)

print(x_test_filtered_vector_recon.shape)
print(x_train_filtered_vector_recon.shape)
print(x_val.shape)

def train_decision_stumps(X_train, y_train, X_val, y_val, n_stumps=300):
    accuracies = []
    num_iteration = []
    n_samples= X_train.shape[0]
    weights = np.ones(n_samples) / n_samples

    best_accuracy = 0
    best_stump = None

    for i in range(n_stumps):
        stump = DecisionStump(X_train, y_train, weights)
        stump.train()
        predictions = stump.predict(X_val)
        mismatch_indices = np.where(predictions != y_val)[0]
        accuracy = len(mismatch_indices) / len(y_val)
        accuracies.append(accuracy)
        num_iteration.append(i + 1)
        weights = stump.get_weights()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_stump = stump
    print("Training done")
    return best_stump, accuracies, num_iteration

best_stump, accuracies, num_iteration = train_decision_stumps(x_train_filtered_vector_recon, y_train_filtered, x_val, y_val)

plt.plot(num_iteration, accuracies)
plt.xlabel('Number of Stumps')
plt.ylabel('Accuracy on Validation Set')
plt.title('Accuracy vs. Number of Stumps')
plt.show()

# Evaluate the best stump on the test set
test_predictions = best_stump.predict(x_test_filtered_vector_recon)
test_accuracy = np.mean(test_predictions == y_test_filtered)
print("Test Accuracy:", test_accuracy)
