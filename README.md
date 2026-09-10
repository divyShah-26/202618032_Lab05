# Applied Statistical Modeling & Interactive Web Dashboard

## M.Sc. Data Science — Semester 1 | Lab 05

**Student:** Divy Shah
**Dataset:** Seaborn Tips Dataset — Restaurant Tipping Behavior
**Technology:** Python, Pandas, NumPy, Seaborn, Matplotlib, SciPy, Statsmodels, Streamlit

---

## 1. Project Overview

This project performs **Exploratory Data Analysis, Hypothesis Testing, Multiple Linear Regression, Model Diagnostics, and Interactive Prediction** using the Seaborn `tips` dataset.

The final analysis is presented through an interactive **Streamlit dashboard:** https://divyshah-26-202618032-lab05-app-njimim.streamlit.app 

### Dataset

The dataset contains **244 restaurant transactions** with information about:

* Total bill
* Tip amount
* Customer sex
* Smoker status
* Day
* Meal time
* Party size

---

## 2. Exploratory Data Analysis

The following analyses were performed:

* Mean, median and standard deviation
* Interquartile Range (IQR)
* Skewness and kurtosis
* Histograms with KDE
* Scatter plots
* Correlation matrix
* Categorical frequency analysis

The analysis showed a positive relationship between **total bill and tip amount**.

---

## 3. Hypothesis Testing

### Test 1 — Two-Group Comparison

Tip amounts were compared between **smokers and non-smokers**.

The analysis included:

* Shapiro-Wilk normality test
* Levene's equal variance test
* Independent t-test / Welch's t-test / Mann-Whitney U test depending on assumptions

At α = 0.05, there was **no statistically significant difference in tip amounts between smokers and non-smokers**.

### Test 2 — Chi-Square Test

A Chi-Square Test of Independence was performed between:

**Smoking Status × Meal Time**

Results:

* χ² = **0.505**
* df = **1**
* p-value = **0.477**

Since p > 0.05, we **fail to reject H₀**.

**Conclusion:** There is insufficient evidence of an association between smoking status and meal time.

---

## 4. Multiple Linear Regression

A multiple linear regression model was created using `statsmodels.api.OLS`.

### Target Variable

`tip`

### Predictors

* `total_bill`
* `size`
* `sex`
* `smoker`
* `day`
* `time`

Categorical variables were converted into dummy variables.

### Model Results

* **R² = 0.470**
* **Adjusted R² = 0.452**
* **F-statistic = 26.06**
* **Model p-value < 0.001**

The model explains approximately **47% of the variation in tip amount**.

### Important Finding

`total_bill` was the only statistically significant individual predictor at the 5% significance level.

Its coefficient was approximately:

**0.0945**

This means that, holding other variables constant, a one-unit increase in total bill is associated with approximately a **0.0945-unit increase in tip**.

---

## 5. Model Diagnostics

The following diagnostic checks were performed:

### Residual Normality

Jarque-Bera:

* JB = **52.555**
* p-value = **3.87 × 10⁻¹²**

The residuals therefore **do not follow a normal distribution**.

### Autocorrelation

Durbin-Watson:

**2.096**

This is close to 2, indicating little evidence of first-order autocorrelation.

### Multicollinearity

VIF was calculated for the continuous predictors:

| Variable   |   VIF |
| ---------- | ----: |
| total_bill | 8.684 |
| size       | 8.684 |

These values indicate **relatively high/potentially concerning multicollinearity**, although they are below 10.

---

## 6. Streamlit Dashboard

The project was converted into an interactive Streamlit application with three tabs.

### Tab 1 — Data Exploration

Includes:

* Interactive sidebar filters
* Dataset summary
* Descriptive statistics
* Histogram + KDE
* Total Bill vs Tip scatter plot
* Correlation heatmap

### Tab 2 — Hypothesis Testing Lab

Includes:

* Dynamic variable selection
* Shapiro-Wilk test
* Levene's test
* Automatic selection of t-test/Welch/Mann-Whitney U
* Chi-Square test
* Test statistics and p-values
* Automatic statistical conclusions

### Tab 3 — Live Prediction & Diagnostics

Includes:

* Interactive customer inputs
* Live tip prediction
* 95% confidence interval
* 95% prediction interval
* R² and adjusted R²
* Residuals vs fitted plot
* Q-Q plot
* Jarque-Bera test
* Durbin-Watson statistic
* VIF analysis

---

## 7. Project Structure

```text
202618032_lab05/
│
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
│
└── data/
    └── tips.csv
```

---

## 8. Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
streamlit
pandas
numpy
seaborn
matplotlib
scipy
statsmodels
plotly
```

Streamlit Community Cloud uses `requirements.txt` to install Python dependencies during deployment.

---

## 9. Run Locally

Activate the virtual environment and run:

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

---

## 10. GitHub Repository

**GitHub:**
https://github.com/divyShah-26/202618032_Lab05

---

## 11. Live Streamlit Application

**Live Dashboard:**


The application is deployed using Streamlit Community Cloud. Community Cloud deploys the application directly from the GitHub repository and uses the repository's dependency file to create the application environment.

---


## 12. Conclusion

The project demonstrates the complete statistical modeling workflow:

**Data Exploration → Hypothesis Testing → Regression Modeling → Diagnostics → Interactive Prediction → Web Deployment**

The analysis shows that **total bill is the strongest significant predictor of tip amount**, while the regression model explains approximately **47% of the variation in tips**.

The final Streamlit dashboard makes the statistical analysis interactive and allows users to explore the dataset, perform hypothesis tests, and generate live predictions.
