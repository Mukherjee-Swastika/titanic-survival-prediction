# Titanic Survival Prediction: A Comparative ML Study

This repository contains a comprehensive machine learning project that predicts passenger survival on the Titanic using the classic Kaggle Titanic dataset. The project implements data preprocessing, feature engineering, categorical encoding, and evaluates five different classification algorithms.

## 📁 Project Structure
* `TitanicModels.ipynb` - Google Colab / Jupyter Notebook containing the full pipeline and model evaluations.
* `README.md` - Project documentation.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn

## 📊 Dataset & Preprocessing
The dataset is loaded directly via Seaborn (`sns.load_dataset('titanic')`). 
### Preprocessing Steps Taken:
1. **Feature Dropping:** Removed redundant or high-cardinality columns (`deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`).
2. **Missing Value Imputation:** Filled missing `age` values with the column mean and dropped rows with missing `embarked` values.
3. **Categorical Encoding:** Leveraged `LabelEncoder` to convert `sex` and `embarked` into numerical formats.
4. **Feature Scaling:** Applied `StandardScaler` for distance-based models (KNN, SVM).

## 📈 Model Performance & Accuracy Comparison

The models were evaluated on an 80/20 train-test split. Below is the detailed breakdown of how each classifier performed:

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.34%** | **0.74** | **0.77** | **0.75** |
| **Support Vector Machine (Linear)** | **80.34%** | **0.74** | **0.75** | **0.75** |
| K-Nearest Neighbors (K=5) | 77.53% | 0.71 | 0.71 | 0.71 |
| Gaussian Naive Bayes | 77.53% | 0.68 | 0.78 | 0.73 |
| Decision Tree Classifier | 76.97% | 0.70 | 0.71 | 0.71 |

### 🏆 Which Model is Most Effective?
Based on the empirical evidence, **Logistic Regression** and **Linear Support Vector Machine (SVM)** tied for the highest overall predictive performance, both achieving an **Accuracy of 80.34%**. 

However, looking closer at the operational metrics:
* **Logistic Regression** edge out slightly as the most balanced model because it achieved a higher **Recall (77%)** for the survival class (Class 1) compared to SVM (75%), meaning it was better at capturing actual survivors while maintaining an identical F1-Score of 0.75.
* To further validate stability, a **5-Fold Cross Validation** was performed on the Linear SVM model, yielding a highly robust and reliable **mean cross-validation accuracy of 78.74%**, proving the model generalizes well and is not overfitted to the test split.

