# Support Vector Machines: Kernel-Based Classification Analysis

## Overview
This project provides a systematic study of Support Vector Machines (SVMs) for
classification tasks, with a focus on understanding how kernel choice, data
characteristics, and hyperparameter tuning influence model behavior and
generalization.

Rather than treating SVM as a black-box classifier, the project emphasizes
conceptual insight and empirical analysis.

## Objective
The main objectives of this project are to:
- study the limitations of linear classifiers on non-linearly separable data,
- analyze the effect of different SVM kernels on decision boundaries,
- understand the role of regularization and hyperparameters in controlling model
  complexity and generalization.

## Approach
The project follows a progressive experimental workflow:
- starting with linearly separable synthetic datasets to establish baseline
  behavior,
- extending to non-linear, noisy, and highly overlapping datasets,
- applying different kernels (linear, polynomial, RBF) and comparing their
  behavior,
- investigating the impact of data preprocessing, particularly feature
  normalization,
- tuning hyperparameters using systematic search strategies.

Both binary and multi-class classification settings are explored.

## Key Concepts
support vector machines • kernel methods • margin maximization •
regularization • hyperparameter tuning • data preprocessing • model robustness

## Datasets
The experiments are conducted on a combination of synthetic datasets and
well-known benchmark datasets commonly used in machine learning research.

## Challenges
Key challenges addressed in this project include:
- selecting appropriate kernels for data with different geometric structures,
- managing sensitivity to feature scaling and data noise,
- balancing model simplicity and flexibility through regularization parameters,
- handling computational cost during hyperparameter optimization.

## How to Run
1. Install the required Python dependencies  
2. Run the main script:
   ```bash
   python main.py
