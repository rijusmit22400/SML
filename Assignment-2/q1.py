import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
with np.load('mnist.npz', allow_pickle=True) as f:
    train_images, train_labels = f['x_train'], f['y_train']
    test_images, test_labels = f['x_test'], f['y_test']

# Visualize 5 samples from each class
fig, axes = plt.subplots(10, 5, figsize=(10, 10))
for i in range(10):
    for j in range(5):
        idx = np.where(train_labels == i)[0][j]
        axes[i, j].imshow(train_images[idx], cmap='gray')
        axes[i, j].axis('off')
plt.show()

# Vectorize images
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Compute mean and covariance for each class
mean_vectors = [] # contains 1 mean for each class: total classes 10
covariance_matrices = [] #contains  elements for each class: total classes 10

for i in range(10):
    class_images = train_images[train_labels == i]
    mean_vectors.append(np.mean(class_images, axis=0))
    covariance_matrices.append(np.cov(class_images, rowvar=False))
mean_vectors=np.array(mean_vectors)
covariance_matrices=np.array(covariance_matrices)

# Compute Prior probabilities for each class
class_counts_train = np.bincount(train_labels, minlength=10)
total_samples_train = len(train_labels)
prior_probabilities = class_counts_train / total_samples_train

classify_samples=[]

# Implement Quadratic Discriminant Analysis (QDA)
def QDA(test_images, mean_vectors, covariance_matrices, prior_probabilities):
    classified_images = []
    for img in test_images:
        scores = []
        for i in range(len(mean_vectors)):
            covariance = covariance_matrices[i] + 1e-6 * np.eye(covariance_matrices[i].shape[0]) # Add a small regularization term
            n_covariance_inverse = np.linalg.pinv(covariance)
            det = np.linalg.slogdet(covariance)
            A = np.dot(np.dot(img.T, n_covariance_inverse), img)
            B = np.dot(np.dot(mean_vectors[i].T, n_covariance_inverse), img)
            C = np.dot(np.dot(mean_vectors[i].T, n_covariance_inverse), mean_vectors[i])
            score = -0.5 * det[0] * det[1] - 0.5 * (A - 2 * B + C) + np.log(prior_probabilities[i])
            scores.append(score)
        classified_images.append(np.argmax(scores))
    return classified_images

# Evaluate accuracy
def evaluate_accuracy(predicted_labels, true_labels):
    total_samples = len(true_labels)
    correct_predictions = np.sum(predicted_labels == true_labels)
    accuracy = correct_predictions / total_samples
    return accuracy


# Classify test images using QDA
predicted_labels = QDA(test_images, mean_vectors, covariance_matrices, prior_probabilities)

accuracy = evaluate_accuracy(predicted_labels, test_labels)
print("Overall Accuracy:", accuracy)

# Class-wise accuracy
class_wise_accuracy = []
for i in range(10):
    class_indices = np.where(test_labels == i)[0]
    class_predicted_labels = predicted_labels[class_indices]
    class_true_labels = test_labels[class_indices]
    class_accuracy = evaluate_accuracy(class_predicted_labels, class_true_labels)
    class_wise_accuracy.append(class_accuracy)
    print(f"Class {i} Accuracy:", class_accuracy) 