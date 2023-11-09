# Fetal-Health-Classification

Project Overview
Cardiotocograms (CTGs) play a crucial role in assessing fetal health by providing valuable information about fetal heart rate (FHR), movements, uterine contractions, and more. This project follows the Knowledge Discovery in Databases (KDD) process, aiming to extract meaningful knowledge from a large dataset of CTG features. The KDD process involves various stages, including data preprocessing, transformation, mining, pattern evaluation, and knowledge representation.

Problem Statement
- The primary goal of this project is to develop a multiclass model capable of classifying CTG features into three fetal health states. The challenge lies in addressing the imbalanced nature of the dataset and ensuring that the model's performance is robust across all classes, particularly the minority classes.

Addressing Imbalanced Data
- To tackle the imbalance in the dataset, a cost-sensitive algorithm was employed. This algorithm penalizes misclassification errors from the minority class more than those from the majority class. This approach helps the model focus on capturing patterns in the underrepresented classes, thus improving overall performance.

StratifiedKFold for Robust Evaluation
- StratifiedKFold was utilized during the cross-validation process to ensure that each fold represents the class distribution of the entire dataset. This step is critical in preventing biased model evaluation, especially when dealing with imbalanced datasets. By preserving the percentage of samples for each class in each fold, the model is assessed in a manner that reflects its ability to generalize to all classes.

Nested Cross Validation for Hyperparameter Tuning
- Nested cross-validation was performed for hyperparameter tuning, employing a combination of an outer loop (StratifiedKFold) and an inner loop (GridSearchCV). This approach provides a robust estimation of the model's performance by iteratively optimizing hyperparameters and evaluating performance across different train-test splits.

Evaluation Metrics
- While accuracy is a commonly used metric, its sensitivity to class distribution makes it insufficient for imbalanced datasets. Therefore, F1 score, which balances precision and recall, was chosen as an additional evaluation metric. F1 score provides a nuanced understanding of the model's performance, especially in capturing the nuances of minority classes. This becomes crucial in scenarios where the class distribution is uneven, preventing accuracy from being misleading.