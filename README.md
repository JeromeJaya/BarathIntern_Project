# Iris Flower Classification

This repository contains a machine learning project focused on classifying Iris flower species based on their sepal and petal characteristics. The project utilizes a Jupyter Notebook (`iris_classification.ipynb`) to walk through the process of data exploration, preprocessing, model training, and evaluation.

## Project Overview

The primary goal of this project is to build a predictive model that can accurately classify an Iris flower into one of three species:

*   Iris-setosa
*   Iris-versicolor
*   Iris-virginica

This is achieved by analyzing the following features:

*   Sepal Length (cm)
*   Sepal Width (cm)
*   Petal Length (cm)
*   Petal Width (cm)

## Notebook Contents

The `iris_classification.ipynb` notebook covers the following key steps:

1.  **Data Loading:**
    *   The Iris dataset is loaded into a pandas DataFrame from the `Iris.csv` file.
2.  **Data Exploration and Preprocessing:**
    *   **Initial Data Inspection:** The notebook starts by examining the basic structure and properties of the dataset using:
        *   `.head()`: To display the first few rows of the data.
        *   `.shape`: To check the dimensions (number of rows and columns).
        *   `.columns`: To list all column names.
        *   `.info()`: To get a concise summary of the DataFrame, including data types and non-null counts.
    *   **Statistical Summary:** `.describe().T` is used to generate descriptive statistics (mean, std, min, max, etc.) for the numerical features, transposed for better readability.
    *   **Null Value Check:** `.isnull().sum()` is employed to count any missing values in each column, ensuring data completeness.
    *   **Feature Removal:** The `Id` column is dropped from the DataFrame as it serves as an identifier and is not useful for the classification task.
3.  **Data Visualization:**
    To better understand the characteristics of the Iris dataset and the relationships between its features, the notebook employs several visualization techniques:
    *   **Histograms:** Histograms are generated for each of the four numerical features (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm). These plots help visualize the distribution of each feature, showing central tendency, spread, and skewness.
    *   **Scatter Plots:** A series of scatter plots are created to explore the relationships between pairs of features:
        *   Sepal Length vs. Sepal Width
        *   Petal Length vs. Petal Width
        *   Sepal Length vs. Petal Length
        *   Sepal Width vs. Petal Width
        Each point in these scatter plots is color-coded by the Iris species. This allows for a visual assessment of how well different feature combinations can distinguish between the three species (Iris-setosa, Iris-versicolor, and Iris-virginica).
    *   **Correlation Heatmap:** A heatmap is generated from the correlation matrix of the features. This provides a visual representation of the linear relationships between pairs of features. Higher positive or negative correlation values are indicated by more intense colors, helping to identify potential multicollinearity or features strongly related to each other.
4.  **Model Training and Evaluation:**
    The notebook proceeds with training and evaluating several machine learning models for the classification task:
    *   **Train-Test Split:** The dataset (with 'Species' as the target `Y` and the other relevant features as `X`) is divided into training and testing sets. This is done using a 70/30 split, meaning 70% of the data is used for training the models and 30% is used for evaluating their performance. The `random_state=42` parameter is used during the split to ensure that the data is divided in the same way each time the code is run, making the results reproducible.
    *   **Models Implemented:** Three different classification algorithms from the scikit-learn library are trained and evaluated:
        *   **Logistic Regression:** A linear model that uses a logistic function to model the probability of a specific class.
        *   **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies a new data point based on the majority class of its 'k' nearest neighbors in the feature space.
        *   **Decision Tree Classifier:** A tree-like model where each internal node represents a decision based on a feature, each branch represents an outcome of that decision, and each leaf node represents a class label.
    *   **Performance Metric:** The primary metric used to evaluate the models is accuracy, which is the proportion of correctly classified samples in the test set. This is calculated using the `.score()` method of the trained models.
    *   **Results:** As documented in the notebook, all three models (Logistic Regression, K-Nearest Neighbors, and Decision Tree Classifier) achieved an accuracy of 100.0% on the test set. While this is an ideal outcome, it's worth noting that such perfect scores on the Iris dataset can sometimes indicate that the dataset is relatively easy to classify or that the `random_state` might have led to a particularly favorable split.
5.  **Model Saving and Loading:**
    *   **Saving the Model:** After training and evaluation, the Decision Tree Classifier model (which, despite all models achieving 100% accuracy, is the one explicitly saved in the notebook) is serialized and saved to a file named `saved_model.sav`. This is accomplished using the `pickle` library, which is a standard Python way to save and load Python objects.
    *   **Loading the Model:** The notebook also demonstrates the process of loading the saved model from the `saved_model.sav` file back into memory using `pickle.load()`.
    *   **Making Predictions with Loaded Model:** Once loaded, the model is used to make predictions on new, unseen data samples (e.g., `[[6.0, 2.2, 4.0, 1.0]]`, `[[4,3,1,5]]`, `[[2,4,8,9]]`) to showcase its ability to classify Iris flower species.
6.  **Conclusion:**
    The `iris_classification.ipynb` notebook serves as a comprehensive yet straightforward example of a complete machine learning classification workflow. It demonstrates all key stages, from initial data loading and thorough exploration through various visualizations, to essential preprocessing steps, model training using different algorithms, performance evaluation, and finally, model persistence (saving and loading) for future application. This makes it a valuable learning resource for understanding the fundamental steps involved in a typical classification task.

## Dataset

The project utilizes the classic Iris dataset, a widely used benchmark in machine learning and statistics. This dataset is loaded from a file named `Iris.csv` within the `iris_classification.ipynb` notebook.

The Iris dataset contains 150 samples of Iris flowers, with 50 samples from each of the following three species:

*   **Iris-setosa**
*   **Iris-versicolor**
*   **Iris-virginica**

For each sample, the following four features are measured:

*   **SepalLengthCm:** The length of the sepal in centimeters.
*   **SepalWidthCm:** The width of the sepal in centimeters.
*   **PetalLengthCm:** The length of the petal in centimeters.
*   **PetalWidthCm:** The width of the petal in centimeters.

The "Id" column, originally present in the `Iris.csv` file, is dropped during the preprocessing stage as it is not relevant for the classification task. The dataset is expected to be clean with no missing values.

## Requirements

The notebook utilizes the following Python libraries:

*   pandas
*   matplotlib
*   numpy
*   seaborn
*   scikit-learn (sklearn)
*   pickle

To run the notebook, ensure you have these libraries installed in your Python environment.

## Usage

1.  Clone this repository.
2.  Ensure you have the `Iris.csv` file in the same directory as the notebook or update the path in the notebook accordingly.
3.  Open and run the `iris_classification.ipynb` notebook in a Jupyter environment.

The notebook will guide you through the classification process, and the trained model will be saved as `save_model.sav`.
