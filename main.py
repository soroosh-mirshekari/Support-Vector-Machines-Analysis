import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_breast_cancer, make_classification, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Helper function to draw decision boundaries to avoid repeating code
def plot_decision_boundary(model, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, alpha=0.8)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Use decision_function for binary classification and predict for multi-class
    if hasattr(model, "decision_function"):
        Z = model.decision_function(xy).reshape(XX.shape)
        # Plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    else:
        Z = model.predict(xy).reshape(XX.shape)
        ax.contourf(XX, YY, Z, cmap='bwr', alpha=0.2)

    plt.title(title)


# Part 1: Basic
print("--- Part 1: Basic ---")

# 1.3: Generate fake data
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1.5: Create and train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 1.6: Predict and calculate accuracy
y_pred = model.predict(X_test)
print(f"Accuracy on blobs: {accuracy_score(y_test, y_pred):.2f}")

# 1.7: Plot decision boundary
plt.figure(figsize=(8, 6))
plot_decision_boundary(model, X, y, "SVM Decision Boundary on Blobs")
plt.show()
#challenge with bad accuracy
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy on circles with linear kernel: {accuracy_score(y_test, y_pred):.2f}")
plt.figure(figsize=(8, 6))
plot_decision_boundary(model, X, y, "Linear SVM on Non-linear Circles Data")
plt.show()


# Challenge: Non-linear data
print("\n--- Part 1 Challenge: Non-linear Data ---")
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy on moons with linear kernel: {accuracy_score(y_test, y_pred):.2f}")
plt.figure(figsize=(8, 6))
plot_decision_boundary(model, X, y, "Linear SVM on Non-linear Moons Data")
plt.show()


# Part 2: Play with Kernel
print("\n--- Part 2: Play with Kernel ---")

# 2.2: Load Iris data
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X = X[y != 0] # Keep only class 1 and 2
y = y[y != 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2.4 & 2.5: Create, train, and evaluate three kernels
kernels = ['linear', 'rbf', 'poly']
models_iris = {}
print("\n--- Iris Dataset Accuracies ---")
for kernel in kernels:
    model = SVC(kernel=kernel, degree=3, gamma='auto')
    model.fit(X_train, y_train)
    models_iris[kernel] = model
    y_pred = model.predict(X_test)
    print(f"{kernel.capitalize()} Kernel Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 2.6: Draw the decision boundary for each model on Iris data
plt.figure(figsize=(18, 5))
for i, (kernel, model) in enumerate(models_iris.items()):
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(model, X, y, f"Iris SVM with {kernel.capitalize()} Kernel")
plt.tight_layout()
plt.show()

# 2.7: Change dataset to breast cancer
print("\n--- Part 2.7: Breast Cancer Dataset ---")
cancer = load_breast_cancer()
X = cancer.data[:, 4:6]
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models_cancer = {}
print("\n--- Breast Cancer Dataset Accuracies ---")
for kernel in kernels:
    model = SVC(kernel=kernel, degree=3, gamma='auto')
    model.fit(X_train, y_train)
    models_cancer[kernel] = model
    y_pred = model.predict(X_test)
    print(f"{kernel.capitalize()} Kernel Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Plot decision boundaries for breast cancer data
plt.figure(figsize=(18, 5))
for i, (kernel, model) in enumerate(models_cancer.items()):
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(model, X, y, f"Cancer SVM with {kernel.capitalize()} Kernel")
plt.tight_layout()
plt.show()

# Part 3: Play with noise (Final and Safest Version)
print("\n--- Part 3: Play with noise ---")

# --- STEP 1: Load original clean data ---
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_clean = cancer.data[:, 4:6]
y_clean = cancer.target
kernels = ['linear', 'rbf', 'poly']
rng = np.random.RandomState(42)

#BLOCK A: Analysis for Noise Level 0.3 
print("\n--- ANALYSIS FOR NOISE LEVEL 0.3 ---")

# Add noise 0.3 to the clean data
noise_03 = rng.normal(0, 1, X_clean.shape)
X_noisy_03 = X_clean + 0.3 * noise_03

# Split the 0.3 noisy data
X_train_03, X_test_03, y_train_03, y_test_03 = train_test_split(
    X_noisy_03, y_clean, test_size=0.3, random_state=42
)

# Train and evaluate models on 0.3 noisy data
models_noisy_03 = {}
print("Accuracy Results for Noise = 0.3:")
for kernel in kernels:
    model = SVC(kernel=kernel, degree=3, gamma='auto')
    model.fit(X_train_03, y_train_03)
    models_noisy_03[kernel] = model
    y_pred = model.predict(X_test_03)
    print(f"  {kernel.capitalize()} Kernel Accuracy: {accuracy_score(y_test_03, y_pred):.4f}")

# Plot decision boundaries for 0.3 noisy data
plt.figure(figsize=(18, 5))
plt.suptitle('Decision Boundaries for Noise Level: 0.3', fontsize=16)
for i, (kernel, model) in enumerate(models_noisy_03.items()):
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(model, X_noisy_03, y_clean, f"SVM with {kernel.capitalize()} Kernel")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#BLOCK B: Analysis for Noise Level 0.5 ---
print("\n--- ANALYSIS FOR NOISE LEVEL 0.5 ---")

# Add noise 0.5 to the clean data
# NOTE: We are generating NEW noise and adding it to the ORIGINAL X_clean
noise_05 = rng.normal(0, 1, X_clean.shape)
X_noisy_05 = X_clean + 0.5 * noise_05

# Split the 0.5 noisy data
X_train_05, X_test_05, y_train_05, y_test_05 = train_test_split(
    X_noisy_05, y_clean, test_size=0.3, random_state=42
)

# Train and evaluate models on 0.5 noisy data
models_noisy_05 = {}
print("Accuracy Results for Noise = 0.5:")
for kernel in kernels:
    model = SVC(kernel=kernel, degree=3, gamma='auto')
    model.fit(X_train_05, y_train_05)
    models_noisy_05[kernel] = model
    y_pred = model.predict(X_test_05)
    print(f"  {kernel.capitalize()} Kernel Accuracy: {accuracy_score(y_test_05, y_pred):.4f}")

# Plot decision boundaries for 0.5 noisy data
plt.figure(figsize=(18, 5))
plt.suptitle('Decision Boundaries for Noise Level: 0.5', fontsize=16)
for i, (kernel, model) in enumerate(models_noisy_05.items()):
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(model, X_noisy_05, y_clean, f"SVM with {kernel.capitalize()} Kernel")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Part 4: Let's be more serious
print("\n--- Part 4: Overlapping Data ---")
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, 
                           class_sep=0.8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models_overlap = {}
print("\n--- Overlapping Dataset Accuracies ---")
for kernel in kernels:
    model = SVC(kernel=kernel, degree=3, gamma='auto')
    model.fit(X_train, y_train)
    models_overlap[kernel] = model
    y_pred = model.predict(X_test)
    print(f"{kernel.capitalize()} Kernel Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Plot decision boundaries for overlapping data
plt.figure(figsize=(18, 5))
for i, (kernel, model) in enumerate(models_overlap.items()):
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(model, X, y, f"Overlapping SVM with {kernel.capitalize()} Kernel")
plt.tight_layout()
plt.show()


# Part 5: Control complexity with C
print("\n--- Part 5: Control complexity with C ---")
X = cancer.data[:, 4:6]
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

C_values = [0.01, 1,5,10, 1000] # Simplified for visualization
models_C = {}
print("\n--- Testing different C values (RBF Kernel) ---")
for c_val in C_values:
    model = SVC(kernel='rbf', C=c_val, gamma='auto')
    model.fit(X_train, y_train)
    models_C[f"C={c_val}"] = model
    y_pred = model.predict(X_test)
    print(f"Accuracy for C={c_val}: {accuracy_score(y_test, y_pred):.4f}")

# Plot decision boundaries for different C values
plt.figure(figsize=(18, 5))
for i, (c_val, model) in enumerate(models_C.items()):
    plt.subplot(1, 5, i + 1)
    plot_decision_boundary(model, X, y, f"RBF Kernel with {c_val}")
plt.tight_layout()
plt.show()

# 5.4: Confusion Matrix for the best model (e.g., C=1)
best_model = models_C['C=1000']
print("\n--- Confusion Matrix for C=1000 ---")
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix for C=1000")
plt.show()


# Part 6: Multi-class SVM model
print("\n--- Part 6: Multi-class SVM ---")
wine = load_wine()
X = wine.data[:, :2] # Using first two features for visualization
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using rbf kernel and OvR strategy as specified
model_wine = SVC(kernel='rbf', decision_function_shape='ovr', gamma='auto')
model_wine.fit(X_train, y_train)
y_pred = model_wine.predict(X_test)
print(f"Accuracy on Wine dataset: {accuracy_score(y_test, y_pred):.2f}")

# Plot decision boundary for multi-class problem
plt.figure(figsize=(8, 6))
# For multi-class, we shade the regions instead of drawing margin lines
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 100)
yy = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_wine.predict(xy).reshape(XX.shape)

plt.contourf(XX, YY, Z, cmap='coolwarm', alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')
plt.title("Multi-class SVM Decision Boundary on Wine Dataset")
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.show()


# Part 7: Bonus Points
print("\n\n\n--- Part 7: Bonus Points ---")

# Bonus 1: Compare SVM with Logistic Regression
print("\n--- Bonus 1: SVM vs. Logistic Regression ---")
cancer = load_breast_cancer()
X = cancer.data[:, :2]  # Use first two features for visualization
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', gamma='auto').fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print(f"SVM (RBF) Accuracy: {accuracy_score(y_test, svm_pred):.4f}")

# Train Logistic Regression
log_reg_model = LogisticRegression().fit(X_train, y_train)
log_reg_pred = log_reg_model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_reg_pred):.4f}")

# Plot boundaries side-by-side
plt.figure(figsize=(14, 6))
# SVM Plot
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, edgecolors='k')
ax1 = plt.gca()
xlim1, ylim1 = ax1.get_xlim(), ax1.get_ylim()
xx1, yy1 = np.linspace(xlim1[0], xlim1[1], 100), np.linspace(ylim1[0], ylim1[1], 100)
YY1, XX1 = np.meshgrid(yy1, xx1)
xy1 = np.vstack([XX1.ravel(), YY1.ravel()]).T
Z1 = svm_model.decision_function(xy1).reshape(XX1.shape)
ax1.contour(XX1, YY1, Z1, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("SVM (RBF) Decision Boundary")

# Logistic Regression Plot
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, edgecolors='k')
ax2 = plt.gca()
xlim2, ylim2 = ax2.get_xlim(), ax2.get_ylim()
xx2, yy2 = np.linspace(xlim2[0], xlim2[1], 100), np.linspace(ylim2[0], ylim2[1], 100)
YY2, XX2 = np.meshgrid(yy2, xx2)
xy2 = np.vstack([XX2.ravel(), YY2.ravel()]).T
Z2 = log_reg_model.predict(xy2).reshape(XX2.shape)
ax2.contourf(XX2, YY2, Z2, cmap='bwr', alpha=0.2)
plt.title("Logistic Regression Decision Boundary")
plt.tight_layout()
plt.show()

# Bonus 2: Effect of Feature Scaling
print("\n--- Bonus 2: Effect of Feature Scaling ---")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model without scaling
model_unscaled = SVC(kernel='rbf', C=1).fit(X_train, y_train)
pred_unscaled = model_unscaled.predict(X_test)
print(f"Accuracy without scaling: {accuracy_score(y_test, pred_unscaled):.4f}")

# Model with scaling, using a Pipeline for convenience
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1))
])
pipeline.fit(X_train, y_train)
pred_scaled = pipeline.predict(X_test)
print(f"Accuracy with StandardScaler: {accuracy_score(y_test, pred_scaled):.4f}")

# Bonus 3: Handling Imbalanced Data
print("\n--- Bonus 3: Handling Imbalanced Data ---")
# Create an imbalanced dataset (90% class 0, 10% class 1)
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, 
                           weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standard model, likely biased towards the majority class
model_imbalanced = SVC(kernel='rbf').fit(X_train, y_train)
pred_imbalanced = model_imbalanced.predict(X_test)
print("--- Results without class_weight ---")
print("(Notice the poor recall for the minority class, 1)")
print(classification_report(y_test, pred_imbalanced))

# Model with balanced class weights
model_balanced = SVC(kernel='rbf', class_weight='balanced').fit(X_train, y_train)
pred_balanced = model_balanced.predict(X_test)
print("\n--- Results with class_weight='balanced' ---")
print("(Recall for the minority class, 1, is now much better)")
print(classification_report(y_test, pred_balanced))

# Bonus 6: GridSearchCV for Hyperparameter Tuning (Optimized for Speed)
print("\n--- Bonus 6: GridSearchCV for Hyperparameter Tuning (Optimized for Speed) ---")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define a slightly smaller parameter grid to make it faster
param_grid = {
    'C': [0.1, 1, 10],           
    'gamma': [0.1, 0.01, 0.001],    
    'kernel': ['rbf'] # focused on rbf
}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1, cv=3)

print("Starting GridSearchCV... This will be much faster now.")
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
grid_pred = grid_search.predict(X_test)
print(f"Test set accuracy with best parameters: {accuracy_score(y_test, grid_pred):.4f}")

# Bonus 7: Sensitivity to Outliers
print("\n--- Bonus 7: Sensitivity to Outliers ---")
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)

# Model on the original, clean data
model_no_outliers = SVC(kernel='linear').fit(X, y)

# Add a single outlier
outlier = np.array([[-2, 6]])
X_with_outlier = np.vstack([X, outlier])
y_with_outlier = np.hstack([y, [0]]) # Assign the outlier to class 0

# Retrain the model on the data with the outlier
model_with_outlier = SVC(kernel='linear').fit(X_with_outlier, y_with_outlier)

# Plotting to visually compare the effect
plt.figure(figsize=(14, 6))
# Plot without outlier
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
ax1 = plt.gca()
xlim1, ylim1 = ax1.get_xlim(), ax1.get_ylim()
xx1, yy1 = np.linspace(xlim1[0], xlim1[1], 30), np.linspace(ylim1[0], ylim1[1], 30)
YY1, XX1 = np.meshgrid(yy1, xx1)
xy1 = np.vstack([XX1.ravel(), YY1.ravel()]).T
Z1 = model_no_outliers.decision_function(xy1).reshape(XX1.shape)
ax1.contour(XX1, YY1, Z1, colors='k', levels=[0], alpha=0.5)
plt.title("SVM without Outlier")

# Plot with outlier
plt.subplot(1, 2, 2)
plt.scatter(X_with_outlier[:, 0], X_with_outlier[:, 1], c=y_with_outlier, cmap='bwr', s=30)
ax2 = plt.gca()
xlim2, ylim2 = ax2.get_xlim(), ax2.get_ylim()
xx2, yy2 = np.linspace(xlim2[0], xlim2[1], 30), np.linspace(ylim2[0], ylim2[1], 30)
YY2, XX2 = np.meshgrid(yy2, xx2)
xy2 = np.vstack([XX2.ravel(), YY2.ravel()]).T
Z2 = model_with_outlier.decision_function(xy2).reshape(XX2.shape)
ax2.contour(XX2, YY2, Z2, colors='k', levels=[0], alpha=0.5)
plt.scatter(outlier[:, 0], outlier[:, 1], c='red', s=150, marker='x', label='Outlier')
plt.title("SVM with Outlier")
plt.legend()
plt.tight_layout()
plt.show()

# Bonus 9: Custom Kernel
print("\n--- Bonus 9: Custom Kernel ---")
# A custom kernel is any function that computes the dot product of two vectors in some feature space
# Here we define a simple linear kernel to demonstrate
def my_custom_linear_kernel(X1, X2):
    """A custom implementation of the linear kernel."""
    return np.dot(X1, X2.T)

# Using the simple Iris data
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X = X[y < 2]; y = y[y < 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train with the built-in linear kernel
model_builtin = SVC(kernel='linear').fit(X_train, y_train)
pred_builtin = model_builtin.predict(X_test)
print(f"Accuracy with built-in linear kernel: {accuracy_score(y_test, pred_builtin):.4f}")

# Train with our custom kernel function
model_custom = SVC(kernel=my_custom_linear_kernel).fit(X_train, y_train)
pred_custom = model_custom.predict(X_test)
print(f"Accuracy with custom linear kernel: {accuracy_score(y_test, pred_custom):.4f}")


