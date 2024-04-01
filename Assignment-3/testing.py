import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

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
pca = PCA(n_components=10)
x_train_pca = pca.fit_transform(x_train_filtered_vector)
x_test_pca = pca.transform(x_test_filtered_vector)

# Learn a decision tree with 3 terminal nodes
decision_tree = DecisionTreeClassifier(max_leaf_nodes=3)
decision_tree.fit(x_train_pca, y_train_filtered)

# Classify test samples using decision tree
decision_tree_predictions = decision_tree.predict(x_test_pca)

# Calculate accuracy and class-wise accuracy for decision tree
decision_tree_accuracy = accuracy_score(y_test_filtered, decision_tree_predictions)
decision_tree_class_wise_accuracy = {}
for c in np.unique(y_test_filtered):
    decision_tree_class_wise_accuracy[c] = accuracy_score(y_test_filtered[y_test_filtered == c],
                                                         decision_tree_predictions[y_test_filtered == c])

print("Decision Tree Accuracy:", decision_tree_accuracy)
print("Decision Tree Class-wise Accuracy:", decision_tree_class_wise_accuracy)

# Now use bagging
bagging_classifier = BaggingClassifier(n_estimators=5)
bagging_classifier.fit(x_train_pca, y_train_filtered)

# Classify test samples using bagging
bagging_predictions = bagging_classifier.predict(x_test_pca)

# Calculate total accuracy and class-wise accuracy for bagging
bagging_accuracy = accuracy_score(y_test_filtered, bagging_predictions)
bagging_class_wise_accuracy = {}
for c in np.unique(y_test_filtered):
    bagging_class_wise_accuracy[c] = accuracy_score(y_test_filtered[y_test_filtered == c],
                                                    bagging_predictions[y_test_filtered == c])

print("Bagging Accuracy:", bagging_accuracy)
print("Bagging Class-wise Accuracy:", bagging_class_wise_accuracy)
