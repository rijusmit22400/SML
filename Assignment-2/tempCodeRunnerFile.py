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

print(np.array(covariance_matrices).shape)
print(len(np.array(mean_vectors)))
print(len(prior_probabilities))

classify_samples=[]

# Implement Quadratic Discriminant Analysis (QDA)
def QDA_EXP(x, mean, covariance, prior_probability):
    n_covariance = covariance + 10e-6 * np.eye(covariance.shape[0])
    n_covariance_inverse = np.linalg.pinv(n_covariance)
    result = -(0.5)*np.log(np.linalg.det(n_covariance)) -(0.5)*(np.dot(np.dot(x.T,n_covariance_inverse),x)-2*np.dot(np.dot(mean.T,n_covariance_inverse),x)+np.dot(np.dot(mean.T,n_covariance_inverse),mean)) + np.log(prior_probability)
    return result

def QDA(x, mean_vectors, covariance_matrices, prior_probabilities):
    scores = []
    for i in range(len(mean_vectors)):
        score = QDA_EXP(x, mean_vectors[i], covariance_matrices[i], prior_probabilities[i])
        scores.append(score)
    return np.argmax(scores)

# Classify test samples
def classify_test_samples(test_images, mean_vectors, covariance_matrices, prior_probabilities):
    predicted_labels = []
    for img in test_images:
        label = QDA(img, mean_vectors, covariance_matrices, prior_probabilities)
        predicted_labels.append(label)
    return np.array(predicted_labels)

# Evaluate accuracy
def evaluate_accuracy(predicted_labels, true_labels):
    total_samples = len(true_labels)
    correct_predictions = np.sum(predicted_labels == true_labels)
    accuracy = correct_predictions / total_samples
    return accuracy

# Classify test samples
predicted_labels = classify_test_samples(test_images, mean_vectors, covariance_matrices, prior_probabilities)

# Evaluate accuracy
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
