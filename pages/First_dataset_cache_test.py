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

# A function that will parse the date Time based cohort:  1 day of month
def get_month(x):
    return dt.datetime(x.year, x.month, 1)


@st.experimental_memo(suppress_st_warning=True)
# @st.experimental_memo
# @st.cache
# @st.cache(suppress_st_warning=True)
def load_data():
    # Loading dataset
    transaction_df = pd.read_excel("transaction.xlsx")  # Load the transaction data
    # transaction_df = pd.read_excel("relay-foods.xlsx", sheet_name=1)
    transaction_df = transaction_df.replace(" ", np.NaN)
    transaction_df = transaction_df.fillna(transaction_df.mean())
    transaction_df["TransactionMonth"] = transaction_df["transaction_date"].apply(
        get_month
    )
    transaction_df["TransactionYear"] = transaction_df["transaction_date"].dt.year
    transaction_df["TransactionMonth"] = transaction_df["transaction_date"].dt.month
    for col in transaction_df.columns:
        if transaction_df[col].dtype == "object":
            transaction_df[col] = transaction_df[col].fillna(
                transaction_df[col].value_counts().index[0]
            )

    # Create transaction_date column based on month and store in TransactionMonth
    transaction_df["TransactionMonth"] = transaction_df["transaction_date"].apply(
        get_month
    )
    # Grouping by customer_id and select the InvoiceMonth value
    grouping = transaction_df.groupby("customer_id")["TransactionMonth"]
    # Assigning a minimum InvoiceMonth value to the dataset
    transaction_df["CohortMonth"] = grouping.transform("min")
    # printing top 5 rows
    # st.write(transaction_df.head())

    # return st.write(transaction_df)
    # return transaction_df
    st.write(transaction_df)


load_data()

