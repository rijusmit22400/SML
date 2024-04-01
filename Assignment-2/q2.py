import numpy as np
import matplotlib.pyplot as plt

# loading the dataset
f = np.load('mnist.npz')
train_ims, train_labels = f['x_train'], f['y_train']
test_ims, test_labels = f['x_test'], f['y_test']

# matrix of 784*1000
selected_ims = []
for i in range(10):
    indices = np.where(train_labels == i)[0][:100]
    for j in indices:
        selected_ims.append(train_ims[j].flatten())

# Convert selected images to numpy array
selected_ims = np.array(selected_ims)

# Subtract the mean from X
mean_of_mat = np.mean(selected_ims, axis=0)
selected_ims_centered = selected_ims - mean_of_mat

# Compute covariance matrix
S = np.dot(selected_ims_centered.T, selected_ims_centered) / 999

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(S)

# Get indices for sorting in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Create matrix U for PCA
U = sorted_eigenvectors

# Perform transformation Y = U^T X
Y = np.dot(selected_ims_centered, U)

# Reconstruct the original data X_recon = UY
X_recon = np.dot(Y, U.T) 

# Calculate Mean Squared Error (MSE)
MSE = np.mean((selected_ims_centered - X_recon)**2)

print("Mean Squared Error (MSE):", MSE)

# Choose values of p & plot 5 images from each class for each value of p
for i in range(1,5):
    if(i==3):
        continue
    p=5*i
    # Select the first p eigenvectors from U
    Up = U[:, :p]
    
    # Perform transformation Y = Up^T X
    Yp = np.dot(selected_ims_centered, Up)
    
    # Reconstruct the original data X_recon = UpYp
    X_recon_p = np.dot(Yp, Up.T) + mean_of_mat
    
    # Reshape each column to 28x28
    X_recon_p = X_recon_p.reshape((-1, 28, 28))
    
    
    # Convert complex values to float and take the real part
    X_recon_p = np.real(X_recon_p)
    
    # Normalize the data to [0, 255]
    X_recon_p = ((X_recon_p - np.min(X_recon_p)) / (np.max(X_recon_p) - np.min(X_recon_p))) * 255
    
    # Plot 5 images from each class
    fig, axes = plt.subplots(10, 5, figsize=(10, 20))
    fig.suptitle(f"Reconstructed Images with p={p}", fontsize=16)
    
    for j in range(10):
        indices = np.where(train_labels == j)[0][:5]
        for k, index in enumerate(indices):
            axes[j, k].imshow(X_recon_p[index], cmap='gray')
            axes[j, k].axis('off')
    
    plt.show()
    
    #using qda to classify samples