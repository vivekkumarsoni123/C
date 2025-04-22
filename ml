Experiment-15:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# Standardize
scaled = StandardScaler().fit_transform(df)
# PCA with 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
# Scatter plot
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data.target, cmap='plasma')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Breast Cancer')
plt.show()
# Heatmap of PCA components
sns.heatmap(pd.DataFrame(pca.components_, columns=data.feature_names), cmap='plasma')
plt.title('PCA Components')
plt.show()

Experiment-13:
import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
# Load and clean dataset
data = pd.read_csv('heart.csv').replace('?', np.nan)
# Define model structure
model = BayesianModel([
    ('age', 'heartdisease'), 
    ('sex', 'heartdisease'), 
    ('exang', 'heartdisease'), 
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'), 
    ('heartdisease', 'chol')
])
# Train model using MLE
model.fit(data, estimator=MaximumLikelihoodEstimator)
# Inference
infer = VariableElimination(model)
# Queries
print("P(HeartDisease | restecg=1)")
print(infer.query(['heartdisease'], evidence={'restecg': 1}))
print("\nP(HeartDisease | cp=2)")
print(infer.query(['heartdisease'], evidence={'cp': 2}))

Experiment-11:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
# Load dataset
names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width', 'Class']
data = pd.read_csv("8-dataset.csv", names=names)
# Input features and labels
X = data.iloc[:, :-1]
labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = [labels[i] for i in data['Class']]
# Color map
colors = np.array(['red', 'lime', 'black'])
# Plot Real Labels
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.title("Actual")
plt.scatter(X.Petal_Length, X.Petal_Width, c=colors[y])
# KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)
plt.subplot(1, 3, 2)
plt.title("KMeans")
plt.scatter(X.Petal_Length, X.Petal_Width, c=colors[kmeans_labels])
print("KMeans Accuracy:", accuracy_score(y, kmeans_labels))
print("KMeans Confusion:\n", confusion_matrix(y, kmeans_labels))
# Gaussian Mixture
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X)
plt.subplot(1, 3, 3)
plt.title("GMM")
plt.scatter(X.Petal_Length, X.Petal_Width, c=colors[gmm_labels])
plt.tight_layout()
plt.show()
print("GMM Accuracy:", accuracy_score(y, gmm_labels))
print("GMM Confusion:\n", confusion_matrix(y, gmm_labels))

Experiment: 9
import numpy as np
import matplotlib.pyplot as plt
from moepy import lowess
# Generate data
x = np.linspace(0, 5, 150)
y = np.sin(x) + np.random.normal(0, 0.1, size=len(x))
# Fit LOWESS model
model = lowess.Lowess()
model.fit(x, y)
# Predict
x_pred = np.linspace(0, 5, 26)
y_pred = model.predict(x_pred)
# Plot
plt.scatter(x, y, color='magenta', s=10, label='Noisy Data')
plt.plot(x_pred, y_pred, 'r--', label='LOWESS Fit')
plt.legend()
plt.show()

Experiment: 7
import numpy as np

# Input and output
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)
X = X / np.max(X, axis=0)
y = y / 100
# Activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return x * (1 - x)
# Parameters
epoch = 7000
lr = 0.1
input_neurons = 2
hidden_neurons = 3
output_neurons = 1
# Weights and biases
wh = np.random.rand(input_neurons, hidden_neurons)
bh = np.random.rand(1, hidden_neurons)
wout = np.random.rand(hidden_neurons, output_neurons)
bout = np.random.rand(1, output_neurons)
# Training
for _ in range(epoch):
    # Forward
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, wout) + bout
    output = sigmoid(final_input)
    # Backward
    error = y - output
    d_output = error * d_sigmoid(output)
    error_hidden = d_output.dot(wout.T)
    d_hidden = error_hidden * d_sigmoid(hidden_output)
        # Update weights
    wout += hidden_output.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr
# Results
print("Input:\n", X)
print("Actual:\n", y)
print("Predicted:\n", output)


Experiment: 5
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# Load dataset
data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target
# Remove duplicates
df = df.drop_duplicates()
# Split data
X = df.drop('TARGET', axis=1)
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Bias: Training error
train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_pred)
# Variance: Testing error
test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_pred)
# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Training Error (Bias):", train_mse)
print("Testing Error (Variance):", test_mse)
print("Cross-Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())


Experiment: 3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
# Load data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)
# Train model
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
# Predict and evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Plot tree
plt.figure(figsize=(5,5))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()


Experiment: 1
import pandas as pd
import numpy as np
data = pd.read_csv("ws.csv")
d = np.array(data)[:,:-1]
target = np.array(data)[:,-1]
def train(c, t):
    for i in range(len(t)):
        if t[i] == "Yes":
            h = c[i].copy()
            break
    for i in range(len(t)):
        if t[i] == "Yes":
            for j in range(len(h)):
                if c[i][j] != h[j]:
                    h[j] = '?'
    return h
print("Final Hypothesis:", train(d, target))
