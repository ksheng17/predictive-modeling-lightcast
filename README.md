# Predictive Modeling of Salary for Lightcast Job Listing Data

This project builds an end-to-end machine learning pipeline using **PySpark** to predict job salaries from job posting data. It demonstrates data cleaning, feature engineering, and model comparison at scale using distributed computing.

The goal is to understand how factors like experience, employment type, and remote status influence compensation, and to evaluate both statistical and machine learning approaches.

---

## Tech Stack:
* **PySpark (Spark MLlib)** – distributed data processing & modeling
* **Python**
* **Pandas / NumPy** – local analysis & transformations
* **Matplotlib / Seaborn** – visualization
* **Plotly** – interactive plotting

---

## Dataset:
* Source: Lightcast job postings dataset
* Format: CSV
* Size: ~49,000 records after cleaning

---

## Data Cleaning & Preprocessing

Key steps:
* Imputed missing values using:
  * Median values
  * Logical rules (e.g., salary ranges)
* Standardized categorical variables:
  * Employment type
  * Remote work type
* Handled inconsistent and null entries
* Selected relevant features for modeling

### Features Used

**Continuous:**
* Minimum years of experience
* Maximum years of experience
* Job posting duration
* Minimum education level

**Categorical:**
* Employment type
* Remote type

**Target:**
* Salary

---

## Feature Engineering

* One-hot encoding for categorical variables using:
  * `StringIndexer`
  * `OneHotEncoder`
* Feature vector assembly via `VectorAssembler`
* Polynomial feature:
  * Squared experience term to capture nonlinearity

---

## Models Implemented

### 1. Generalized Linear Regression (GLR)
* Baseline linear model
* Interpretable coefficients, p-values, and statistical metrics

### 2. Polynomial Regression
* Adds nonlinear relationship via squared experience
* Slight performance improvement over linear model

### 3. Random Forest Regressor

* Ensemble model capturing nonlinear interactions
* Best performing model in this project
