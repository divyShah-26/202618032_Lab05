import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Restaurant Tips Statistical Dashboard",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================

DATA_PATH = Path("data/tips.csv")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    df = sns.load_dataset("tips")


# =========================================================
# TITLE
# =========================================================

st.title("🍽️ Restaurant Tipping Behavior Dashboard")

st.markdown(
    """
    This interactive dashboard performs exploratory data analysis,
    hypothesis testing, multiple linear regression, prediction,
    and diagnostic analysis using the Seaborn **Tips** dataset.
    """
)


# =========================================================
# PREPARE DATA
# =========================================================

numerical_columns = [
    "total_bill",
    "tip",
    "size"
]

categorical_columns = [
    "sex",
    "smoker",
    "day",
    "time"
]


# =========================================================
# BUILD REGRESSION MODEL
# =========================================================

y = df["tip"].astype(float)

X_raw = df[
    [
        "total_bill",
        "size",
        "sex",
        "smoker",
        "day",
        "time"
    ]
]

X = pd.get_dummies(
    X_raw,
    drop_first=True,
    dtype=float
)

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔎 Data Filters")

bill_min = float(df["total_bill"].min())
bill_max = float(df["total_bill"].max())

bill_range = st.sidebar.slider(
    "Total Bill Range",
    min_value=bill_min,
    max_value=bill_max,
    value=(bill_min, bill_max)
)

selected_day = st.sidebar.multiselect(
    "Day",
    options=sorted(df["day"].dropna().unique()),
    default=sorted(df["day"].dropna().unique())
)

selected_smoker = st.sidebar.multiselect(
    "Smoker",
    options=sorted(df["smoker"].dropna().unique()),
    default=sorted(df["smoker"].dropna().unique())
)

selected_time = st.sidebar.multiselect(
    "Meal Time",
    options=sorted(df["time"].dropna().unique()),
    default=sorted(df["time"].dropna().unique())
)


# Apply filters

filtered_df = df[
    (df["total_bill"] >= bill_range[0]) &
    (df["total_bill"] <= bill_range[1]) &
    (df["day"].isin(selected_day)) &
    (df["smoker"].isin(selected_smoker)) &
    (df["time"].isin(selected_time))
]


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Data Exploration",
        "🧪 Hypothesis Testing Lab",
        "🔮 Live Prediction & Diagnostics"
    ]
)


# =========================================================
# TAB 1 — DATA EXPLORATION
# =========================================================

with tab1:

    st.header("📊 Data Exploration")

    # -----------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        filtered_df.shape[0]
    )

    col2.metric(
        "Columns",
        filtered_df.shape[1]
    )

    col3.metric(
        "Average Bill",
        f"${filtered_df['total_bill'].mean():.2f}"
        if len(filtered_df) > 0 else "N/A"
    )

    col4.metric(
        "Average Tip",
        f"${filtered_df['tip'].mean():.2f}"
        if len(filtered_df) > 0 else "N/A"
    )

    st.dataframe(
        filtered_df.head(10),
        use_container_width=True
    )


    # -----------------------------------------------------
    # Descriptive statistics
    # -----------------------------------------------------

    st.subheader("Descriptive Statistics")

    if len(filtered_df) > 0:

        descriptive_table = pd.DataFrame({
            "Mean": filtered_df[numerical_columns].mean(),
            "Median": filtered_df[numerical_columns].median(),
            "Std Dev": filtered_df[numerical_columns].std(),
            "IQR": (
                filtered_df[numerical_columns].quantile(0.75)
                - filtered_df[numerical_columns].quantile(0.25)
            ),
            "Skewness": filtered_df[numerical_columns].skew(),
            "Kurtosis": filtered_df[numerical_columns].kurt()
        })

        st.dataframe(
            descriptive_table.round(3),
            use_container_width=True
        )


    # -----------------------------------------------------
    # Histogram
    # -----------------------------------------------------

    st.subheader("Distribution of Total Bill")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=filtered_df,
        x="total_bill",
        kde=True,
        ax=ax
    )

    ax.set_title("Distribution of Total Bill")
    ax.set_xlabel("Total Bill ($)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)


    # -----------------------------------------------------
    # Scatter plot
    # -----------------------------------------------------

    st.subheader("Total Bill vs Tip")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=filtered_df,
        x="total_bill",
        y="tip",
        hue="smoker",
        ax=ax
    )

    ax.set_title("Total Bill vs Tip")
    ax.set_xlabel("Total Bill ($)")
    ax.set_ylabel("Tip ($)")

    st.pyplot(fig)


    # -----------------------------------------------------
    # Correlation matrix
    # -----------------------------------------------------

    st.subheader("Correlation Matrix")

    correlation_matrix = filtered_df[numerical_columns].corr()

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    st.pyplot(fig)


# =========================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# =========================================================

with tab2:

    st.header("🧪 Hypothesis Testing Lab")


    # =====================================================
    # TEST 1 — TWO GROUP COMPARISON
    # =====================================================

    st.subheader("Test 1: Two-Group Comparison")

    st.markdown(
        """
        Select a numerical variable and a categorical variable
        containing exactly two groups.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        numerical_variable = st.selectbox(
            "Numerical Variable",
            numerical_columns
        )

    with col2:

        group_variable = st.selectbox(
            "Grouping Variable",
            ["sex", "smoker", "time"]
        )


    groups = df[group_variable].dropna().unique()


    if len(groups) == 2:

        group1 = df[
            df[group_variable] == groups[0]
        ][numerical_variable].dropna()

        group2 = df[
            df[group_variable] == groups[1]
        ][numerical_variable].dropna()


        st.write(
            f"**Groups:** {groups[0]} vs {groups[1]}"
        )


        # -------------------------------------------------
        # Shapiro-Wilk
        # -------------------------------------------------

        shapiro1 = stats.shapiro(group1)
        shapiro2 = stats.shapiro(group2)

        normal1 = shapiro1.pvalue > 0.05
        normal2 = shapiro2.pvalue > 0.05


        # -------------------------------------------------
        # Levene
        # -------------------------------------------------

        levene_result = stats.levene(
            group1,
            group2
        )

        equal_variance = (
            levene_result.pvalue > 0.05
        )


        # -------------------------------------------------
        # Display assumptions
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Shapiro p-value",
            f"{shapiro1.pvalue:.4f}"
        )

        col2.metric(
            "Shapiro p-value",
            f"{shapiro2.pvalue:.4f}"
        )

        col3.metric(
            "Levene p-value",
            f"{levene_result.pvalue:.4f}"
        )


        # -------------------------------------------------
        # Select statistical test
        # -------------------------------------------------

        if normal1 and normal2:

            if equal_variance:

                test_result = stats.ttest_ind(
                    group1,
                    group2,
                    equal_var=True
                )

                test_name = (
                    "Independent Two-Sample t-test"
                )

            else:

                test_result = stats.ttest_ind(
                    group1,
                    group2,
                    equal_var=False
                )

                test_name = "Welch's t-test"

        else:

            test_result = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U test"


        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.subheader("Test Result")

        st.write(
            f"**Test Used:** {test_name}"
        )

        st.write(
            f"**Test Statistic:** "
            f"{test_result.statistic:.4f}"
        )

        st.write(
            f"**p-value:** "
            f"{test_result.pvalue:.6f}"
        )


        # -------------------------------------------------
        # Hypotheses
        # -------------------------------------------------

        st.markdown("### Hypotheses")

        st.write(
            "H₀: There is no statistically significant "
            "difference between the two groups."
        )

        st.write(
            "H₁: There is a statistically significant "
            "difference between the two groups."
        )


        # -------------------------------------------------
        # Conclusion
        # -------------------------------------------------

        if test_result.pvalue < 0.05:

            st.error(
                "Reject H₀: There is a statistically "
                "significant difference between the groups."
            )

        else:

            st.success(
                "Fail to Reject H₀: There is no statistically "
                "significant difference between the groups."
            )


    # =====================================================
    # TEST 2 — CHI-SQUARE
    # =====================================================

    st.divider()

    st.subheader(
        "Test 2: Chi-Square Test of Independence"
    )

    col1, col2 = st.columns(2)

    with col1:

        categorical_1 = st.selectbox(
            "First Categorical Variable",
            categorical_columns,
            index=1
        )

    with col2:

        categorical_2 = st.selectbox(
            "Second Categorical Variable",
            categorical_columns,
            index=3
        )


    if categorical_1 != categorical_2:

        contingency_table = pd.crosstab(
            df[categorical_1],
            df[categorical_2]
        )

        chi2, p_value, dof, expected = (
            stats.chi2_contingency(
                contingency_table
            )
        )


        st.write("### Contingency Table")

        st.dataframe(
            contingency_table,
            use_container_width=True
        )


        st.write("### Expected Frequencies")

        expected_table = pd.DataFrame(
            expected,
            index=contingency_table.index,
            columns=contingency_table.columns
        )

        st.dataframe(
            expected_table.round(2),
            use_container_width=True
        )


        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Chi-Square Statistic",
            f"{chi2:.4f}"
        )

        col2.metric(
            "Degrees of Freedom",
            dof
        )

        col3.metric(
            "p-value",
            f"{p_value:.6f}"
        )


        st.markdown("### Hypotheses")

        st.write(
            "H₀: The two categorical variables are independent."
        )

        st.write(
            "H₁: The two categorical variables are associated."
        )


        if p_value < 0.05:

            st.error(
                "Reject H₀: There is a statistically significant "
                "association between the categorical variables."
            )

        else:

            st.success(
                "Fail to Reject H₀: There is insufficient evidence "
                "of an association between the categorical variables."
            )

    else:

        st.warning(
            "Please select two different categorical variables."
        )


# =========================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# =========================================================

with tab3:

    st.header(
        "🔮 Live Prediction & Diagnostics"
    )


    # =====================================================
    # LIVE PREDICTION
    # =====================================================

    st.subheader("Live Tip Prediction")

    col1, col2 = st.columns(2)

    with col1:

        prediction_bill = st.number_input(
            "Total Bill ($)",
            min_value=float(df["total_bill"].min()),
            max_value=float(df["total_bill"].max()),
            value=20.0,
            step=0.50
        )

        prediction_size = st.number_input(
            "Party Size",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        prediction_sex = st.selectbox(
            "Sex",
            sorted(df["sex"].unique())
        )

    with col2:

        prediction_smoker = st.selectbox(
            "Smoker",
            sorted(df["smoker"].unique())
        )

        prediction_day = st.selectbox(
            "Day",
            sorted(df["day"].unique())
        )

        prediction_time = st.selectbox(
            "Meal Time",
            sorted(df["time"].unique())
        )


    # -----------------------------------------------------
    # Create prediction row
    # -----------------------------------------------------

    new_data = pd.DataFrame({
        "total_bill": [prediction_bill],
        "size": [prediction_size],
        "sex": [prediction_sex],
        "smoker": [prediction_smoker],
        "day": [prediction_day],
        "time": [prediction_time]
    })


    new_X = pd.get_dummies(
        new_data,
        drop_first=True,
        dtype=float
    )

    new_X = sm.add_constant(
        new_X,
        has_constant="add"
    )

    # Match training model columns exactly

    new_X = new_X.reindex(
        columns=X.columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.get_prediction(
        new_X
    )

    prediction_summary = prediction.summary_frame(
        alpha=0.05
    )

    predicted_tip = prediction_summary[
        "mean"
    ].iloc[0]

    confidence_lower = prediction_summary[
        "mean_ci_lower"
    ].iloc[0]

    confidence_upper = prediction_summary[
        "mean_ci_upper"
    ].iloc[0]

    prediction_lower = prediction_summary[
        "obs_ci_lower"
    ].iloc[0]

    prediction_upper = prediction_summary[
        "obs_ci_upper"
    ].iloc[0]


    # -----------------------------------------------------
    # Display prediction
    # -----------------------------------------------------

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Tip",
        f"${predicted_tip:.2f}"
    )

    col2.metric(
        "95% Confidence Interval",
        f"${confidence_lower:.2f} – "
        f"${confidence_upper:.2f}"
    )

    col3.metric(
        "95% Prediction Interval",
        f"${prediction_lower:.2f} – "
        f"${prediction_upper:.2f}"
    )


    st.info(
        "The confidence interval estimates the uncertainty "
        "around the mean predicted tip, while the prediction "
        "interval gives a wider range for an individual future tip."
    )


    # =====================================================
    # REGRESSION MODEL SUMMARY
    # =====================================================

    st.divider()

    st.subheader("Regression Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "R²",
        f"{model.rsquared:.3f}"
    )

    col2.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.3f}"
    )

    col3.metric(
        "F-statistic",
        f"{model.fvalue:.2f}"
    )

    col4.metric(
        "Model p-value",
        f"{model.f_pvalue:.2e}"
    )


    # =====================================================
    # RESIDUAL DIAGNOSTICS
    # =====================================================

    st.divider()

    st.subheader("Residual Diagnostics")

    residuals = model.resid
    fitted_values = model.fittedvalues


    # -----------------------------------------------------
    # Residuals vs Fitted
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        ax=ax
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(
        "Residuals vs Fitted Values"
    )

    st.pyplot(fig)


    # -----------------------------------------------------
    # Q-Q Plot
    # -----------------------------------------------------

    fig = plt.figure(
        figsize=(7, 6)
    )

    sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    plt.title(
        "Q-Q Plot of Regression Residuals"
    )

    st.pyplot(fig)


    # =====================================================
    # JARQUE-BERA
    # =====================================================

    from statsmodels.stats.stattools import (
        jarque_bera,
        durbin_watson
    )

    jb_stat, jb_pvalue, skewness, kurtosis = (
        jarque_bera(residuals)
    )

    dw_stat = durbin_watson(
        residuals
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jarque-Bera Statistic",
            f"{jb_stat:.3f}"
        )

        st.metric(
            "Jarque-Bera p-value",
            f"{jb_pvalue:.2e}"
        )

    with col2:

        st.metric(
            "Durbin-Watson",
            f"{dw_stat:.3f}"
        )

        st.metric(
            "Residual Skewness",
            f"{skewness:.3f}"
        )


    if jb_pvalue < 0.05:

        st.warning(
            "Jarque-Bera test indicates that the residuals "
            "are not normally distributed."
        )

    else:

        st.success(
            "Jarque-Bera test indicates that the residuals "
            "are approximately normally distributed."
        )


    # =====================================================
    # VIF
    # =====================================================

    st.subheader(
        "Variance Inflation Factor (VIF)"
    )

    continuous_vars = X[
        ["total_bill", "size"]
    ]

    vif_data = pd.DataFrame()

    vif_data["Variable"] = (
        continuous_vars.columns
    )

    vif_data["VIF"] = [
        variance_inflation_factor(
            continuous_vars.values,
            i
        )
        for i in range(
            continuous_vars.shape[1]
        )
    ]

    st.dataframe(
        vif_data.round(3),
        use_container_width=True
    )


    st.caption(
        "VIF values above 5 may indicate potentially "
        "concerning multicollinearity."
    )