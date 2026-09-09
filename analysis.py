import pandas as pd
import seaborn as sns

# Load the Restaurant Tips dataset
df = sns.load_dataset("tips")

# Display the first 5 rows
print("===== FIRST 5 ROWS =====")
print(df.head())

# Display dataset dimensions
print("\n===== DATASET SHAPE =====")
print(df.shape)

# Display column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# Display data types and non-null information
print("\n===== DATASET INFORMATION =====")
print(df.info())

# Save a local copy of the dataset
df.to_csv("data/tips.csv", index=False)

import pandas as pd
import seaborn as sns

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = sns.load_dataset("tips")

# Save a local copy
df.to_csv("data/tips.csv", index=False)

print("Dataset loaded successfully!")
print("Dataset saved to data/tips.csv")


# ==========================================
# 2. BASIC DATASET INFORMATION
# ==========================================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())


# ==========================================
# 3. DESCRIPTIVE STATISTICS
# ==========================================

numerical_columns = ["total_bill", "tip", "size"]

print("\n===== DESCRIPTIVE STATISTICS =====")

for column in numerical_columns:

    mean = df[column].mean()
    median = df[column].median()
    std = df[column].std()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    skewness = df[column].skew()
    kurtosis = df[column].kurt()

    print(f"\nVariable: {column}")
    print(f"Mean           : {mean:.3f}")
    print(f"Median         : {median:.3f}")
    print(f"Standard Dev.  : {std:.3f}")
    print(f"Q1             : {q1:.3f}")
    print(f"Q3             : {q3:.3f}")
    print(f"IQR            : {iqr:.3f}")
    print(f"Skewness       : {skewness:.3f}")
    print(f"Kurtosis       : {kurtosis:.3f}")

    # ==========================================
# 4. DESCRIPTIVE STATISTICS TABLE
# ==========================================

descriptive_table = pd.DataFrame({
    "Mean": df[numerical_columns].mean(),
    "Median": df[numerical_columns].median(),
    "Std Dev": df[numerical_columns].std(),
    "IQR": df[numerical_columns].quantile(0.75)
             - df[numerical_columns].quantile(0.25),
    "Skewness": df[numerical_columns].skew(),
    "Kurtosis": df[numerical_columns].kurt()
})

print("\n===== DESCRIPTIVE STATISTICS TABLE =====")
print(descriptive_table.round(3))

# ==========================================
# 5. CATEGORICAL VARIABLE COUNTS
# ==========================================

categorical_columns = ["sex", "smoker", "day", "time"]

print("\n===== CATEGORICAL VARIABLE COUNTS =====")

for column in categorical_columns:
    print(f"\n{column.upper()}")
    print(df[column].value_counts())

# ==========================================
# 6. HISTOGRAM WITH KDE
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="total_bill",
    kde=True
)

plt.title("Distribution of Total Bill")
plt.xlabel("Total Bill ($)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==========================================
# 7. SCATTER PLOT
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")

plt.tight_layout()
plt.show()

# ==========================================
# 8. CORRELATION MATRIX
# ==========================================

correlation_matrix = df[numerical_columns].corr()

print("\n===== CORRELATION MATRIX =====")
print(correlation_matrix.round(3))

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# ==========================================
# 9. HYPOTHESIS TEST 1
# Smokers vs Non-Smokers
# ==========================================

from scipy import stats

# Separate tip values into two groups
smokers = df[df["smoker"] == "Yes"]["tip"]
non_smokers = df[df["smoker"] == "No"]["tip"]

print("\n===== HYPOTHESIS TEST 1 =====")
print("Comparing tip amounts between smokers and non-smokers")

print("\nNumber of smokers:", len(smokers))
print("Number of non-smokers:", len(non_smokers))

print("\nMean tip - Smokers:", smokers.mean())
print("Mean tip - Non-smokers:", non_smokers.mean())

print("\nMedian tip - Smokers:", smokers.median())
print("Median tip - Non-smokers:", non_smokers.median())

# ==========================================
# 10. SHAPIRO-WILK NORMALITY TEST
# ==========================================

shapiro_smokers = stats.shapiro(smokers)
shapiro_non_smokers = stats.shapiro(non_smokers)

print("\n===== SHAPIRO-WILK NORMALITY TEST =====")

print("\nSmokers:")
print("Statistic:", shapiro_smokers.statistic)
print("p-value :", shapiro_smokers.pvalue)

print("\nNon-Smokers:")
print("Statistic:", shapiro_non_smokers.statistic)
print("p-value :", shapiro_non_smokers.pvalue)

# ==========================================
# 11. LEVENE'S TEST FOR EQUAL VARIANCE
# ==========================================

levene_test = stats.levene(smokers, non_smokers)

print("\n===== LEVENE'S TEST =====")
print("Statistic:", levene_test.statistic)
print("p-value :", levene_test.pvalue)

# ==========================================
# 12. SELECT APPROPRIATE STATISTICAL TEST
# ==========================================

alpha = 0.05

normal_smokers = shapiro_smokers.pvalue > alpha
normal_non_smokers = shapiro_non_smokers.pvalue > alpha

print("\n===== TEST SELECTION =====")

if normal_smokers and normal_non_smokers:

    print("Both groups satisfy the normality assumption.")

    equal_variance = levene_test.pvalue > alpha

    if equal_variance:
        print("Equal variance assumption is satisfied.")
        print("Using independent two-sample t-test.")

        test_result = stats.ttest_ind(
            smokers,
            non_smokers,
            equal_var=True
        )

        test_name = "Independent Two-Sample t-test"

    else:
        print("Equal variance assumption is NOT satisfied.")
        print("Using Welch's t-test.")

        test_result = stats.ttest_ind(
            smokers,
            non_smokers,
            equal_var=False
        )

        test_name = "Welch's t-test"

else:

    print("Normality assumption is NOT satisfied.")
    print("Using Mann-Whitney U test.")

    test_result = stats.mannwhitneyu(
        smokers,
        non_smokers,
        alternative="two-sided"
    )

    test_name = "Mann-Whitney U test"


# ==========================================
# 13. FINAL HYPOTHESIS TEST RESULT
# ==========================================

print("\n===== FINAL HYPOTHESIS TEST RESULT =====")

print("Test used:", test_name)
print("Test statistic:", test_result.statistic)
print("p-value:", test_result.pvalue)

if test_result.pvalue < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is a statistically significant difference")
    print("in tip amounts between smokers and non-smokers.")

else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant difference")
    print("in tip amounts between smokers and non-smokers.")

# ==========================================
# 14. HYPOTHESIS TEST 2
# CHI-SQUARE TEST OF INDEPENDENCE
# ==========================================

print("\n===== HYPOTHESIS TEST 2 =====")
print("Chi-Square Test: Smoker vs Meal Time")


# Create contingency table
contingency_table = pd.crosstab(
    df["smoker"],
    df["time"]
)

print("\n===== CONTINGENCY TABLE =====")
print(contingency_table)

# Perform Chi-Square test
chi2, p_value, degrees_of_freedom, expected = stats.chi2_contingency(
    contingency_table
)

print("\n===== CHI-SQUARE TEST RESULT =====")

print("Chi-Square statistic:", chi2)
print("Degrees of freedom:", degrees_of_freedom)
print("p-value:", p_value)

print("\n===== EXPECTED FREQUENCIES =====")

expected_table = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)

print(expected_table.round(2))

# ==========================================
# 15. CHI-SQUARE DECISION
# ==========================================

alpha = 0.05

print("\n===== CHI-SQUARE CONCLUSION =====")

if p_value < alpha:
    print("Decision: Reject H0")
    print(
        "Conclusion: Smoking status and meal time "
        "are significantly associated."
    )
else:
    print("Decision: Fail to Reject H0")
    print(
        "Conclusion: There is insufficient evidence "
        "of an association between smoking status and meal time."
    )

    # ==========================================
# PART 2: MULTIPLE LINEAR REGRESSION
# ==========================================

import statsmodels.api as sm

# Target variable
y = df["tip"].astype(float)

# Predictor variables
X = df[["total_bill", "size", "sex", "smoker", "day", "time"]]

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True, dtype=float)

# Add intercept/constant
X = sm.add_constant(X)

print("\nRegression Predictor Columns:")
print(X.columns)

print("\nFirst 5 rows of Regression Data:")
print(X.head())

# Fit Multiple Linear Regression using OLS
model = sm.OLS(y, X).fit()

# Display regression summary
print("\n==========================================")
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("==========================================")

print(model.summary())

# ==========================================
# PART 2: RESIDUAL DIAGNOSTICS
# ==========================================

residuals = model.resid
fitted_values = model.fittedvalues

print("\n==========================================")
print("RESIDUAL DIAGNOSTICS")
print("==========================================")

print("\nFirst 5 Residuals:")
print(residuals.head())

print("\nFirst 5 Fitted Values:")
print(fitted_values.head())


# ==========================================
# 1. RESIDUALS VS FITTED VALUES
# Linearity & Homoscedasticity
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=fitted_values,
    y=residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.tight_layout()
plt.show()


# ==========================================
# 2. Q-Q PLOT
# Normality of Residuals
# ==========================================

plt.figure(figsize=(7, 6))

sm.qqplot(
    residuals,
    line="45",
    fit=True
)

plt.title("Q-Q Plot of Regression Residuals")

plt.tight_layout()
plt.show()


# ==========================================
# 3. VIF
# Multicollinearity
# ==========================================

from statsmodels.stats.outliers_influence import variance_inflation_factor

continuous_vars = X[["total_bill", "size"]]

vif_data = pd.DataFrame()

vif_data["Variable"] = continuous_vars.columns

vif_data["VIF"] = [
    variance_inflation_factor(
        continuous_vars.values,
        i
    )
    for i in range(continuous_vars.shape[1])
]

print("\n==========================================")
print("VARIANCE INFLATION FACTOR (VIF)")
print("==========================================")

print(vif_data.round(3))


# ==========================================
# 4. DIAGNOSTIC SUMMARY
# ==========================================

from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.stattools import durbin_watson

jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(residuals)
dw_stat = durbin_watson(residuals)

print("\n==========================================")
print("DIAGNOSTIC SUMMARY")
print("==========================================")

print("Jarque-Bera Statistic:", round(jb_stat, 3))
print("Jarque-Bera p-value:", jb_pvalue)
print("Durbin-Watson:", round(dw_stat, 3))

if jb_pvalue < 0.05:
    print("Normality: Residuals are NOT normally distributed.")
else:
    print("Normality: Residuals are approximately normally distributed.")

    
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================================
# MULTICOLLINEARITY - VIF
# ==========================================

continuous_vars = X[['total_bill', 'size']]

vif_data = pd.DataFrame()
vif_data["Variable"] = continuous_vars.columns
vif_data["VIF"] = [
    variance_inflation_factor(continuous_vars.values, i)
    for i in range(continuous_vars.shape[1])
]

print("\n==========================================")
print("VARIANCE INFLATION FACTOR (VIF)")
print("==========================================")
print(vif_data)