import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt

# The code below is for the title and logo.
st.set_page_config(page_title="Cohort Analysis App", page_icon="👥")

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/busts-in-silhouette_1f465.png",
    width=120,
)


st.title("Cohort Analysis App")

st.write("")
st.markdown(
    """This  🎈Streamlit demo is based on the [following code](https://github.com/maladeep/cohort-retention-rate-analysis-in-python).

It calculates the `retention rate` (the percentage of active customers compared to the total number of customers, split by month). This `retention rate` is then visualized and interpreted through a heatmap, powered by [Plotly](https://plotly.com/python/getting-started/).
"""
)

st.write("")
st.write("")

# Loading dataset
transaction_df = pd.read_excel("transaction.xlsx")

with st.expander("Show the `Transactions` dataframe"):
    # st.write(df)
    st.write(transaction_df)


# Inspect missing values in the dataset
# st.write(transaction_df.isnull().values.sum())
# Replace the ' 's with NaN
transaction_df = transaction_df.replace(" ", np.NaN)
# Impute the missing values with mean imputation
transaction_df = transaction_df.fillna(transaction_df.mean())
# Count the number of NaNs in the dataset to verify
# st.write(transaction_df.isnull().values.sum())

# st.write(transaction_df.info())

for col in transaction_df.columns:
    # Check if the column is of object type
    if transaction_df[col].dtypes == "object":
        # Impute with the most frequent value
        transaction_df[col] = transaction_df[col].fillna(
            transaction_df[col].value_counts().index[0]
        )
# Count the number of NaNs in the dataset and print the counts to verify
# st.write(transaction_df.isnull().values.sum())

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
# st.write(transaction_df.head())


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

# Extract the difference in months from all previous values "+1" in addeded at the end so that first month is marked as 1 instead of 0 for easier interpretation. """


transaction_df["CohortIndex"] = years_diff * 12 + months_diff + 1
# st.write(transaction_df.head(5))


with st.form("my_form"):

    col1, col2 = st.columns(2)

    with col1:
        list_price_slider = st.slider("list price (in $)", min_value=12, max_value=2091)
        # list_price_slider = st.slider("list_price", min_value=transaction_df["list_price"].min(), max_value=transaction_df["list_price"].max())

    with col2:
        standard_cost_slider = st.slider("standard cost (in $)", min_value=7, max_value=1759)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:

        # selecting rows based on condition
        transaction_df = transaction_df[
            transaction_df["list_price"] > list_price_slider
        ]
        transaction_df = transaction_df[
            transaction_df["list_price"] > standard_cost_slider
        ]


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
# cohort_data.head()

cohort_sizes = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_sizes, axis=0)
# Coverting the retention rate into percentage and Rounding off.
retention = retention.round(3) * 100
retention.index = retention.index.strftime("%Y-%m")

#############

# st.subheader("Monthly Cohorts for Average Standard Cost")

# st.subheader("Monthly cohorts showing customer retention rates")

fig = go.Figure()

fig.add_heatmap(
    x=retention.columns, y=retention.index, z=retention, colorscale="cividis"
)

fig.layout.title = "Monthly cohorts showing customer retention rates"
fig["layout"]["title"]["font"] = dict(size=25)
fig.layout.template = "none"
fig.layout.width = 750
fig.layout.height = 750
fig.layout.xaxis.tickvals = retention.columns
fig.layout.yaxis.tickvals = retention.index
fig.layout.margin.b = 100
fig
