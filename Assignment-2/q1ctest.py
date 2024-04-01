import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST dataset
mnist = np.load('mnist.npz')
x_train, y_train = mnist['x_train'], mnist['y_train']
x_test, y_test = mnist['x_test'], mnist['y_test']

# Visualize 5 samples from each class in the train set
num_classes = 10
samples_per_class = 5

fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 10))
for i in range(num_classes):
    class_samples = x_train[y_train == i][:samples_per_class]
    for j in range(samples_per_class):
        axes[i, j].imshow(class_samples[j], cmap='gray')
        axes[i, j].axis('off')
plt.show()

# Vectorize images
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Apply Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(x_train, y_train)

# Compute class-wise mean and covariance
class_means = qda.means_
class_covariances = qda.covariance_

# Define QDA expression
def qda_predict(x, class_mean, class_cov):
    num_classes = class_mean.shape[0]
    log_probs = np.zeros(num_classes)
    for i in range(num_classes):
        diff = x - class_mean[i]
        cov_inv = np.linalg.inv(class_cov[i])
        log_probs[i] = -0.5 * diff.T @ cov_inv @ diff - 0.5 * np.log(np.linalg.det(class_cov[i]))
    return np.argmax(log_probs)

# Predict classes for test set
y_pred = [qda_predict(sample, class_means, class_covariances) for sample in x_test]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
class_wise_accuracy = accuracy_score(y_test, y_pred, normalize=False) / np.bincount(y_test)

print("Overall accuracy:", accuracy)
print("Class-wise accuracy:")
for i in range(num_classes):
    print("Class {}: {:.2%}".format(i, class_wise_accuracy[i]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST dataset
mnist = np.load('mnist.npz')
x_train, y_train = mnist['x_train'], mnist['y_train']
x_test, y_test = mnist['x_test'], mnist['y_test']

# Visualize 5 samples from each class in the train set
num_classes = 10
samples_per_class = 5

fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 10))
for i in range(num_classes):
    class_samples = x_train[y_train == i][:samples_per_class]
    for j in range(samples_per_class):
        axes[i, j].imshow(class_samples[j], cmap='gray')
        axes[i, j].axis('off')
plt.show()

# Vectorize images
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Apply Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(x_train, y_train)

# Compute class-wise mean and covariance
class_means = qda.means_
class_covariances = qda.covariance_

# Define QDA expression
def qda_predict(x, class_mean, class_cov):
    num_classes = class_mean.shape[0]
    log_probs = np.zeros(num_classes)
    for i in range(num_classes):
        diff = x - class_mean[i]
        cov_inv = np.linalg.inv(class_cov[i])
        log_probs[i] = -0.5 * diff.T @ cov_inv @ diff - 0.5 * np.log(np.linalg.det(class_cov[i]))
    return np.argmax(log_probs)

# Predict classes for test set
y_pred = [qda_predict(sample, class_means, class_covariances) for sample in x_test]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
class_wise_accuracy = accuracy_score(y_test, y_pred, normalize=False) / np.bincount(y_test)

print("Overall accuracy:", accuracy)
print("Class-wise accuracy:")
for i in range(num_classes):
    print("Class {}: {:.2%}".format(i, class_wise_accuracy[i]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
