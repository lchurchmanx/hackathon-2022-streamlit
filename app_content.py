import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app_variables import *


def load_csv(path):
    dataset = pd.read_csv(os.path.join(path))
    return dataset


def load_parquet(path):
    return pd.read_parquet(path)


def set_home():
    st.write(intro, unsafe_allow_html=True)
    st.write(intro_context, unsafe_allow_html=True)


def preview_data():
    st.title('Data Preview')

    st.markdown('### DataFrame `Ranking`')
    ds = load_parquet(path_ranking_dataset)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Bank Data`')
    st.markdown(
        '')
    dataset = load_csv(path_bank_dataset)
    st.markdown(f'{len(dataset.index)} entries  |  {len(dataset.columns)} columns')
    st.write(dataset.astype(str))

    ds = load_csv(path_merchant_dataset)
    st.markdown('### DataFrame `Merchant Data`')
    st.markdown(
        '')
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))


def in_progress():
    st.markdown('# In Progress...')


def loan_scoring():

    st.markdown('# Loan Score')

    dataset = load_parquet(path_ranking_dataset)

    # print(dataset.columns)
    # Index(['MCC', 'MERCHANT_ID', 'ZIPCODE', 'BILLING_STATE', 'CITY', 'POPULATION',
    #        'MERCHANT_NAME', 'MCC_CATEGORY', 'MCC_DESC', 'TOT_SALES2021',
    #        'TOT_SALES2020', 'TOT_SALES2019', 'TOT_SALE_CNT2021',
    #        'TOT_SALE_CNT2020', 'TOT_SALES_CNT2019', 'DIFF_AMT', 'DIFF_CNT',
    #        'AVG_SALES', 'SALES_TREND_1', 'SALES_TREND_2', 'AVG_SALES_CNT',
    #        'SALES_CNT_TREND_1', 'SALES_CNT_TREND_2', 'SALES_TREND_WEIGHTED',
    #        'SALE_CNT_TREND_WEIGHTED', 'TREND_WEIGHTED', 'TREND_RANKING'],
    #       dtype='object')

    col1, col2, col3, col4, col5 = st.columns((.1, 1, .1, 1, .1))

    with col2:
        state = [
            st.selectbox('state',
                         [x for x in dataset.BILLING_STATE.unique()],
                         key='billing_state')]
        df = dataset.query(f'BILLING_STATE == "{state[0]}"')

        city = [
            st.selectbox('city',
                         [x for x in sorted(df.CITY.unique())],
                         key='city')
        ]
        df = df.query(f'CITY == "{city[0]}"')

        merchant = [
            st.selectbox('merchant',
                         [x for x in sorted(df.MERCHANT_NAME.unique())],
                         key='merchant')
        ]

        loan = [
            st.selectbox('loan',
                         ['Less than 20K', '20- 50K', '50K-120K'],
                         key='loan')
        ]

        df = df.query(f'MERCHANT_NAME == "{merchant[0]}"')
        print(df.TREND_WEIGHTED)
        score = 'TBD'

    with col4:
        st.write('')
    st.markdown(f'### Results')
    # st.markdown(f'Rank _{rank.TREND_RANKING[0]}_')
    # st.markdown(f'Score _{score}_')
    st.write(df.astype(str))
    st.write('')
    for x in df.itertuples():
        st.write(f"Weighted trend for Merchant ID **{x.MERCHANT_ID}**: {x.TREND_WEIGHTED}")
        # Low/Medium / Good

def determine_score(val):
    if val < 0:
        image = Image.open('')
    elif val < 50:
        image = Image.open('')