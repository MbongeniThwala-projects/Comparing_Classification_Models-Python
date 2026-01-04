# Comparing_Classification_Models-Python
This project implements a comprehensive comparison of 14 different machine learning classifiers on a synthetic binary classification dataset. The goal is to evaluate and visualise the performance of various classification algorithms to identify the most effective model for a given task.

🎯 Key Objectives
Generate a synthetic binary classification dataset

Train and evaluate 14 different classification models

Compare model performance using accuracy metrics

Visualize results through both tabular and graphical representations

📈 Dataset Details
Type: Synthetic binary classification dataset

Samples: 1,000

Features: 5

Classes: 2

Redundancy: 0

Split: 80/20 train-test ratio

🤖 Models Evaluated
The project compares the following 14 classifiers:

Nearest_Neighbors (KNeighborsClassifier)

Linear_SVM (SVC with linear kernel)

Polynomial_SVM (SVC with poly kernel)

RBF_SVM (SVC with RBF kernel)

Gaussian_Process (GaussianProcessClassifier)

Gradient_Boosting (GradientBoostingClassifier)

Decision_Tree (DecisionTreeClassifier)

Extra_Trees (ExtraTreesClassifier)

Random_Forest (RandomForestClassifier)

Neural_Net (MLPClassifier)

AdaBoost (AdaBoostClassifier)

Naive_Bayes (GaussianNB)

QDA (QuadraticDiscriminantAnalysis)

SGD (SGDClassifier)

📊 Performance Results
Models are evaluated based on test set accuracy scores ranging from 0.79 to 0.85:

Model	Accuracy Score
Naive_Bayes	0.850
Polynomial_SVM	0.840
Neural_Net	0.840
Multiple models	0.835
Random_Forest	0.825
AdaBoost	0.830
Gradient_Boosting	0.790
Extra_Trees	0.790
Best Performing Model: Naive_Bayes with 85% accuracy

📊 Visualization Features
Color-coded DataFrame: Uses Seaborn's light palette to create a gradient visualization where darker greens indicate higher performance

Horizontal Bar Plot: Displays model performance comparisons using a clean, whitegrid-style bar chart with the viridis palette

🛠️ Technologies Used
Python Libraries:

Scikit-learn (for models and dataset generation)

Pandas (for data manipulation)

Seaborn (for visualization)

NumPy (implicitly through Scikit-learn)

Jupyter Notebook for interactive development

🔑 Key Insights
The Gaussian Naive Bayes classifier performed best on this synthetic dataset

Most models performed similarly with scores between 0.79-0.85

Ensemble methods (Random Forest, Gradient Boosting, Extra Trees) showed competitive but not superior performance

The visualization approach effectively highlights performance differences through both color gradients and bar plots

📁 Project Structure
text
├── Data Generation
│   └── Synthetic dataset with 1000 samples, 5 features
├── Data Splitting
│   └── 80/20 train-test split
├── Model Implementation
│   └── 14 classifiers with standardized hyperparameters
├── Performance Evaluation
│   └── Accuracy scoring on test set
└── Results Visualization
    ├── Color-coded performance table
    └── Comparative bar plot
🚀 How to Run
Clone the repository

Install required packages: pip install scikit-learn pandas seaborn

Open and run the Jupyter Notebook

🎯 Future Improvements
Add cross-validation for more robust evaluation

Include additional performance metrics (precision, recall, F1-score)

Experiment with hyperparameter tuning

Test on real-world datasets for practical validation

Add confusion matrix visualizations for each model
