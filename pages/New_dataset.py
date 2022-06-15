import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objs as go


st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/busts-in-silhouette_1f465.png",
    width=120,
)

st.title("Cohort Analysis App [New Dataset]")

st.write("")
st.markdown(
    """This  🎈Streamlit demo is based on the [following code](https://github.com/maladeep/cohort-retention-rate-analysis-in-python).

It calculates the `retention rate` (the percentage of active customers compared to the total number of customers, split by month). This `retention rate` is then visualized and interpreted through a heatmap, powered by [Plotly](https://plotly.com/python/getting-started/).
"""
)

st.info(
    f"""
        You can review the [dataset here](https://github.com/CharlyWargnier/cohort-analysis-test/blob/main/relay-foods.xlsx)
        """
)

st.write("")
st.write("")

# pd.set_option("max_columns", 50)
# mpl.rcParams["lines.linewidth"] = 2

df = pd.read_excel("relay-foods.xlsx", sheet_name=1)

df["OrderPeriod"] = df.OrderDate.apply(lambda x: x.strftime("%Y-%m"))
df.set_index("UserId", inplace=True)

df["CohortGroup"] = (
    df.groupby(level=0)["OrderDate"].min().apply(lambda x: x.strftime("%Y-%m"))
)
df.reset_index(inplace=True)
df.head()


grouped = df.groupby(["CohortGroup", "OrderPeriod"])

# count the unique users, orders, and total revenue per Group + Period
cohorts = grouped.agg(
    {"UserId": pd.Series.nunique, "OrderId": pd.Series.nunique, "TotalCharges": np.sum}
)

# make the column names more meaningful
cohorts.rename(columns={"UserId": "TotalUsers", "OrderId": "TotalOrders"}, inplace=True)
cohorts.head()


def cohort_period(df):
    """
    Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.

    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['UserId', 'OrderTime', inplace=True)
        df = df.groupby('UserId').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df["CohortPeriod"] = np.arange(len(df)) + 1
    return df


TotalCharges_slider = st.slider(
    "Total Charges (in $)", step=50, min_value=2, max_value=690
)

cohorts = cohorts[cohorts["TotalCharges"] > TotalCharges_slider]

cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts.head()

# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(["CohortGroup", "CohortPeriod"], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts["TotalUsers"].groupby(level=0).first()
cohort_group_size.head()

user_retention = cohorts["TotalUsers"].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)

user_retention[["2009-06", "2009-07", "2009-08"]].plot(figsize=(10, 5))
plt.title("Cohorts: User Retention")
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel("% of Cohort Purchasing")
cohorts["TotalUsers"].head()

user_retention = cohorts["TotalUsers"].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)

user_retention[["2009-06", "2009-07", "2009-08"]].plot(figsize=(10, 5))
plt.title("Cohorts: User Retention")
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel("% of Cohort Purchasing")

# Creating heatmaps in matplotlib is more difficult than it should be.
# Thankfully, Seaborn makes them easy for us.
# http://stanford.edu/~mwaskom/software/seaborn/

# transaction_df_new = df[
#     ["brand", "product_line", "list_price", "standard_cost"]
# ]

# transaction_df_new = df[
#     ["TotalCharges"]
# ]
# new = [col for col in transaction_df_new]


import seaborn as sns

sns.set(style="white")

plt.figure(figsize=(12, 8))
plt.title("Cohorts: User Retention")
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt=".0%")

st.pyplot()

fig = go.Figure()

fig.add_heatmap(
    x=user_retention.columns,
    y=user_retention.index,
    z=user_retention,
    colorscale="cividis",
)

fig.layout.title = "Monthly cohorts showing customer retention rates"
fig["layout"]["title"]["font"] = dict(size=25)
fig.layout.template = "none"
fig.layout.width = 750
fig.layout.height = 750
fig.layout.xaxis.tickvals = user_retention.columns
fig.layout.yaxis.tickvals = user_retention.index
fig.layout.margin.b = 100
fig


st.stop()


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

dtypes = transaction_df.dtypes.astype(str)
# Show dtypes
# dtypes

transaction_df_new = transaction_df[
    ["brand", "product_line", "list_price", "standard_cost"]
]
new = [col for col in transaction_df_new]
# show filtered dataframe (currently with 3 columns)
# new

# with st.form("my_form"):

st.write("")


cole, col1, cole, col2, cole = st.columns([0.1, 1, 0.05, 1, 0.1])


with col1:
    MetricSlider01 = st.selectbox("Pick your 1st metric", new)

    MetricSlider02 = st.selectbox("Pick your 2nd metric", new, index=2)

    st.write("")


with col2:

    if MetricSlider01 == "brand":
        # col_one_list = transaction_df_new["brand"].tolist()
        col_one_list = transaction_df_new["brand"].drop_duplicates().tolist()
        multiselect = st.multiselect(
            "Select the value(s)", col_one_list, ["Solex", "Trek Bicycles"]
        )
        transaction_df = transaction_df[transaction_df["brand"].isin(multiselect)]

    elif MetricSlider01 == "product_line":
        col_one_list = transaction_df_new["product_line"].drop_duplicates().tolist()
        multiselect = st.multiselect(
            "Select the value(s)", col_one_list, ["Standard", "Road"]
        )
        transaction_df = transaction_df[
            transaction_df["product_line"].isin(multiselect)
        ]

    elif MetricSlider01 == "list_price":
        list_price_slider = st.slider(
            "List price (in $)", step=500, min_value=12, max_value=2091
        )
        transaction_df = transaction_df[
            transaction_df["list_price"] > list_price_slider
        ]

    elif MetricSlider01 == "standard_cost":
        TotalCharges_slider = st.slider(
            "Standard cost (in $)", step=500, min_value=7, max_value=1759
        )
        transaction_df = transaction_df[
            transaction_df["list_price"] > TotalCharges_slider
        ]

    if MetricSlider02 == "brand":
        # col_one_list = transaction_df_new["brand"].tolist()
        col_one_list = transaction_df_new["brand"].drop_duplicates().tolist()
        multiselect_02 = st.multiselect(
            "Select the value(s)", col_one_list, ["Solex", "Trek Bicycles"], key=1
        )
        transaction_df = transaction_df[transaction_df["brand"].isin(multiselect)]

    elif MetricSlider02 == "product_line":
        col_one_list = transaction_df_new["product_line"].drop_duplicates().tolist()
        multiselect_02 = st.multiselect(
            "Select the value(s)", col_one_list, ["Standard", "Road"]
        )
        transaction_df = transaction_df[
            transaction_df["product_line"].isin(multiselect)
        ]

    elif MetricSlider02 == "list_price":
        list_price_slider = st.slider(
            "List price (in $)", step=500, min_value=12, max_value=2091
        )
        transaction_df = transaction_df[
            transaction_df["list_price"] > list_price_slider
        ]

    elif MetricSlider02 == "standard_cost":
        TotalCharges_slider = st.slider(
            "Standard cost (in $)", step=500, min_value=7, max_value=1759
        )
        transaction_df = transaction_df[
            transaction_df["list_price"] > TotalCharges_slider
        ]


try:

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

except IndexError:
    st.warning("This is throwing an exception, bear with us!")
