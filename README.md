# Student Grade Prediction using Machine Learning

## Project Overview
This project aims to predict student final grades (**G3**) using various socio-economic and academic factors from the dataset. A **Random Forest** model was trained to analyze the impact of different features on student performance and make accurate predictions.

## Dataset
The dataset includes various attributes related to students, such as:
- **Demographic Information**: Age, sex, address type (urban/rural)
- **Family Background**: Parents' education, family size, and support systems
- **Academic Performance**: Previous grades (G1, G2), study time, failures
- **Extracurricular Activities**: Participation in activities, romantic relationships, alcohol consumption

## Data Preprocessing
- Categorical variables were encoded appropriately.
- Missing values were handled using imputation techniques.
- The dataset was split into **training (80%)** and **testing (20%)** sets.

## Model Training & Evaluation
A **Random Forest Regressor** was used with the following configuration:
- **Number of trees**: 100
- **Feature importance analysis enabled**

### Performance Metrics
The trained model was evaluated using:
- **Root Mean Squared Error (RMSE)**: `1.83`
- **Mean Absolute Error (MAE)**: `1.11`

These metrics indicate that the model provides reasonably accurate predictions, with an average error of about **1.1 grade points**.

## Feature Importance
Key features influencing student performance based on **increase in node purity**:
1. **G2 (Second period grade)** - Most influential feature
2. **G1 (First period grade)**
3. **Absences** - High number of absences negatively affects performance
4. **Failures** - Past academic failures are a strong predictor of future performance
5. **Mjob (Mother's job)** - Indirectly influences academic success

A **feature importance plot** was generated to visualize these results.

## Actual vs. Predicted Scores
A scatter plot was created to compare actual and predicted grades, with a **red dashed line** representing the ideal case where predictions perfectly match actual values.

## How to Run the Project
1. Install necessary libraries:
   ```bash
   install.packages("randomForest")
   install.packages("ggplot2")
   ```
2. Load and preprocess the dataset.
3. Run the Random Forest model and evaluate its performance.
4. Generate feature importance and actual vs. predicted plots.

