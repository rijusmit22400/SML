import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Load MNIST dataset
mnist = np.load('mnist.npz')
X_train, y_train = mnist['x_train'], mnist['y_train']
X_test, y_test = mnist['x_test'], mnist['y_test']

# Visualize 5 samples from each class in the train set
classes = np.unique(y_train)
samples_per_class = 5

plt.figure(figsize=(10, 5))
for i, cls in enumerate(classes):
    idxs = np.where(y_train == cls)[0][:samples_per_class]
    for j, idx in enumerate(idxs):
        plt.subplot(samples_per_class, len(classes), j * len(classes) + i + 1)
        plt.imshow(X_train[idx], cmap='gray')
        plt.axis('off')
        if j == 0:
            plt.title(str(cls))
plt.tight_layout()
plt.show()

# Vectorize images
X_train_vec = X_train.reshape(-1, 28*28)
X_test_vec = X_test.reshape(-1, 28*28)

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(X_train_vec, y_train)

# Compute class-wise mean and covariance vectors
class_means = qda.means_
class_covariances = qda.covariance_

# Compute predictions on test set
y_pred = qda.predict(X_test_vec)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", accuracy)

# Compute class-wise accuracy
conf_mat = confusion_matrix(y_test, y_pred)
class_wise_accuracy = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
print("Class-wise Accuracy:", class_wise_accuracy)
