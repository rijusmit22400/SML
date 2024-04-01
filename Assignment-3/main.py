import numpy as np
import matplotlib.pyplot as plt


# Load the npz file
data = np.load('C:/Users/rijus/OneDrive/Desktop/Projects/SML_Assignements and projects/Assignment-3/mnist.npz')

# Access the arrays within the npz file
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

train_filter = np.isin(y_train, [0, 1, 2])
x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]

test_filter = np.isin(y_test, [0, 1, 2])
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

# Reshape the images to 784*18623
x_train_filtered_vector = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
x_test_filtered_vector = x_test_filtered.reshape(x_test_filtered.shape[0], -1)

#covariance matrix for training data
mean_train = np.mean(x_train_filtered_vector, axis=0)
centered_train = x_train_filtered_vector - mean_train
cov_train = np.dot(centered_train.T, centered_train) / 18622
#eignen values and vectors
eigen_values, eigen_vectors = np.linalg.eig(cov_train)

#sorting the eigen values and vectors
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_indices]
sorted_eigenvectors = eigen_vectors[:, sorted_indices]

#applying the PCA
# Create matrix U for PCA
U = sorted_eigenvectors[0:10]

# Perform transformation Y = U^T X
Y = np.dot(centered_train, U.T)

# Reconstruct the original data X_recon = UY
x_train_filtered_vector_recon = Y @ U

print(x_train_filtered_vector_recon.shape)


def plot_images(images, labels, num_samples=5):
    fig, axes = plt.subplots(len(np.unique(labels)), num_samples, figsize=(10, 6))
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            img_idx = np.random.choice(np.where(labels == i)[0])
            ax.imshow(images[img_idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Class {i}")
    plt.tight_layout()
    plt.show()

# Plot 5 samples from each class
plot_images(x_train_filtered, y_train_filtered)
# Calculating prior probabilites for all clases
init_class=0
prior_prob=[]
for i in range(3):
    prior_prob.append(np.sum(y_train_filtered==i)/len(y_train_filtered))
print(prior_prob)
#total Gini index
total_gini_index=0
for i in range(3):
    total_gini_index+=prior_prob[i]*(1-prior_prob[i])
# Close the file
data.close()