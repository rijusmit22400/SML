import numpy as np

class DecisionStump:
    def __init__(self, train_set, label_data, weights=None):
        self.feature_data = train_set
        self.label_data = label_data
        if weights is None:
            self.weights = np.ones(len(label_data)) / len(label_data)
        else:
            self.weights = weights
        self.split_value = None
        self.alpha = None
    
    def train(self):
        for feature in self.train_set:
            feature_copy = np.unique(feature)
            feature_copy.sort()
            thresholds = (feature_copy[1:] + feature_copy[:-1]) / 2
            min_loss = float('inf')
            best_threshold = None

            for threshold in thresholds:
                predictions = np.where(feature <= threshold, -1, 1)
                misclassified = np.sum(self.weights[predictions != self.label_data])
                loss = misclassified / np.sum(self.weights)
                
                if loss < min_loss:
                    min_loss = loss
                    best_threshold = threshold

            self.split_value = best_threshold
            init_predictions = np.where(feature <= thresholds[0], -1, 1)
            
            #finding missclassified points
            misclassified = np.sum(self.weights[init_predictions != self.label_data])
            
            #loss function
            L = misclassified/np.sum(self.weights)
            
            #alpha value
            a = np.log((1-misclassified)/(misclassified))*(1/2)
            
            #new weights
            self.weights = self.weights*np.exp(-a*init_predictions != self.label_data) 
    def predict(self, X):
        return np.where(X <= self.split_value, -1, 1)

