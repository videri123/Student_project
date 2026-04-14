# Student Performance MLOps Project

This project demonstrates how to **correlate two focus features (`Hours_Studied` and `Attendance`) with the target (`Exam_Score`)** and then reuse the same dataset for regression modeling, evaluation, and explainability.

## Structure

- `StudentPerformanceFactors.csv`: the provided dataset with both numeric and categorical predictors and `Exam_Score` as the target.
- `student_performance_mlops.py`: the runnable Python script that covers the entire workflow.

## Workflow explained

1. **Exploration & correlation**  
   - The script loads the CSV, drops duplicates, and prints missing-value counts.  
   - It computes the correlation matrix for `Hours_Studied`, `Attendance`, and `Exam_Score`, then saves a heatmap and pairplot that visually prove the two focus columns are positively correlated with the target.  
2. **Preprocessing**  
   - Numeric features (including the two focus columns) are standardized with `StandardScaler`.  
   - Categorical features are one-hot encoded (first level dropped) so the regressors can use them without introducing dummy-variable traps.
3. **Modeling & evaluation**  
   - Two pipelines are trained: a `LinearRegression` pipeline for interpretability and a `RandomForestRegressor` pipeline for nonlinear behavior.  
   - Both pipelines share the same preprocessor, keeping the correlation story intact. Cross-validation (5-fold) measures RMSE for each.
4. **Explainability**  
   - After fitting, the script extracts feature importance (coefficients for the linear model or feature importances for the tree), so you can see how `Hours_Studied` and `Attendance` dominate the predictions.  
   - It also runs permutation importance and saves the top 15 feature importance rankings; the same CSV helps emphasize that the two focus columns remain among the most important even after adding dozens of features.
5. **Reporting**  
   - Predicted-versus-actual scatter plots for each model show how well the predictions align with the exam scores.  
   - A short summary file (`analysis_summary.txt`) logs the correlation values and RMSEs to highlight how the correlation analysis and prediction results match.

## Running the project

1. Create a virtual environment (optional but recommended):  
   ```sh
   python -m venv .venv
   .venv\\Scripts\\activate
   ```
2. Install dependencies:  
   ```sh
   pip install pandas seaborn matplotlib scikit-learn
   ```
3. Run the script from the `Student_project` directory:  
   ```sh
   python student_performance_mlops.py
   ```
   This will create an `outputs/` folder with:
   - `correlation_focus_heatmap.png`
   - `focus_pairplot.png`
   - Per-model permutation importance CSVs (e.g., `linearregression_permutation_importance.csv`)
   - Per-model prediction scatter plots
   - `analysis_summary.txt`

4. Inspect the summary file and plots to understand how `Hours_Studied` and `Attendance` correlate with the exam score and how the models learned the same signal.

## Next steps (optional)

1. Add logging (MLflow or Weights & Biases) to track dataset versions, model parameters, and metrics as you iterate.  
2. Extend the script with command-line arguments for hyperparameters or dataset filtering.  
3. Integrate a CI/CD job that reruns this pipeline when the CSV changes, so the correlation story is always documented with fresh plots.
