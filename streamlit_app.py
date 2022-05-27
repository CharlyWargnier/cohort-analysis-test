import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import missingno as msno
from textwrap import wrap

st.set_option("deprecation.showPyplotGlobalUse", False)

# Loading dataset
transaction_df = pd.read_excel("transaction.xlsx")
# View data
st.write(transaction_df.head())


# Inspect missing values in the dataset
st.write(transaction_df.isnull().values.sum())
# Replace the ' 's with NaN
transaction_df = transaction_df.replace(" ", np.NaN)
# Impute the missing values with mean imputation
transaction_df = transaction_df.fillna(transaction_df.mean())
# Count the number of NaNs in the dataset to verify
st.write(transaction_df.isnull().values.sum())


st.write(transaction_df.info())
for col in transaction_df.columns:
    # Check if the column is of object type
    if transaction_df[col].dtypes == "object":
        # Impute with the most frequent value
        transaction_df[col] = transaction_df[col].fillna(
            transaction_df[col].value_counts().index[0]
        )
# Count the number of NaNs in the dataset and print the counts to verify
st.write(transaction_df.isnull().values.sum())


# A function that will parse the date Time based cohort:  1 day of month
def get_month(x):
    return dt.datetime(x.year, x.month, 1)


# Create transaction_date column based on month and store in TransactionMonth
transaction_df["TransactionMonth"] = transaction_df["transaction_date"].apply(get_month)
# Grouping by customer_id and select the InvoiceMonth value
grouping = transaction_df.groupby("customer_id")["TransactionMonth"]
# Assigning a minimum InvoiceMonth value to the dataset
transaction_df["CohortMonth"] = grouping.transform("min")
# printing top 5 rows
st.write(transaction_df.head())


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


# Getting the integers for date parts from the `InvoiceDay` column
transcation_year, transaction_month, _ = get_date_int(
    transaction_df, "TransactionMonth"
)
# Getting the integers for date parts from the `CohortDay` column
cohort_year, cohort_month, _ = get_date_int(transaction_df, "CohortMonth")


#  Get the  difference in years
years_diff = transcation_year - cohort_year
# Calculate difference in months
months_diff = transaction_month - cohort_month
""" Extract the difference in months from all previous values
 "+1" in addeded at the end so that first month is marked as 1 instead of 0 for easier interpretation. 
 """
transaction_df["CohortIndex"] = years_diff * 12 + months_diff + 1
st.write(transaction_df.head(5))

# Counting daily active user from each chort
grouping = transaction_df.groupby(["CohortMonth", "CohortIndex"])
# Counting number of unique customer Id's falling in each group of CohortMonth and CohortIndex
cohort_data = grouping["customer_id"].apply(pd.Series.nunique)
cohort_data = cohort_data.reset_index()
# Assigning column names to the dataframe created above
cohort_counts = cohort_data.pivot(
    index="CohortMonth", columns="CohortIndex", values="customer_id"
)
# Printing top 5 rows of Dataframe
cohort_data.head()

cohort_sizes = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_sizes, axis=0)
# Coverting the retention rate into percentage and Rounding off.
retention.round(3) * 100

retention.index = retention.index.strftime("%Y-%m")


import plotly.graph_objs as go

fig = go.Figure()

fig.add_heatmap(
    x=retention.columns, y=retention.index, z=retention, colorscale="cividis"
)
fig.layout.title = "Average Standard Cost: Monthly Cohorts"

fig.layout.template = "none"

fig.layout.width = 650

fig.layout.height = 650

fig.layout.xaxis.tickvals = retention.columns

fig.layout.yaxis.tickvals = retention.index

fig.layout.margin.b = 100

fig
