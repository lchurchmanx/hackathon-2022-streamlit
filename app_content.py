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

    image = Image.open('images/Low.png')
    st.image(image, width=250)
    image = Image.open('images/Medium.png')
    st.image(image, width=250)
    image = Image.open('images/High.png')
    st.image(image, width=250)


def preview_data():
    st.title('Data Preview')

    st.markdown('### DataFrame `Scoring`')
    ds = load_csv(path_scoring_dataset)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Ranking`')
    ds = load_parquet(path_ranking_dataset)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Merchant Modeling`')
    ds = load_parquet(path_modeling_merchant_dataset)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Merged Data`')
    ds = load_parquet(path_merge_data_dataset)
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

    # dataset = load_parquet(path_ranking_dataset)
    dataset = load_csv(path_scoring_dataset)

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

        # merchant_df = load_csv(path_merchant_dataset)
        # ids = [np.int64(x.MERCHANT_ID) for x in df.itertuples()]
        # filter1 = merchant_df["MERCHANT_ID"].isin(ids)
        # merchant_df = merchant_df[filter1]
        # st.write(merchant_df.astype(str))
        # print(merchant_df)

        # print(merchant_df.dtypes)

        # score_df = load_csv(path_scoring_dataset)
        # filter1 = score_df["MERCHANT_ID"].isin(['4445090630228', '4445028703157'])
        # score_df = score_df[filter1]
        # st.write('another')
        # st.write(score_df.astype(str))
        # #
        # score_df = load_csv(path_merchant_dataset)
        # print("TEST")
        # print(ids)
        # print([4445028768606])
        # filter1 = score_df["MERCHANT_ID"].isin(ids) # takeb from file
        # filter2 = score_df["MERCHANT_ID"].isin([4445028768606]) # takeb from file
        #
        # st.write('FILTER1')
        # score_df1 = score_df[filter1]
        # st.write(score_df1.astype(str))
        #
        # st.write('FILTER2')
        # score_df2 = score_df[filter2]
        # st.write(score_df2.astype(str))

        # print(merchant_df)
        # print(merchant_df.MERCHANT_NAME.unique())

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

        # df = df.query(f'MERCHANT_NAME == "{merchant[0]}"')
        score = 'TBD'

    #     score_df = load_csv(path_scoring_dataset)
    #     ids = [str(x.MERCHANT_ID) for x in df.itertuples()]
    #     # score_df = score_df.query(f'MERCHANT_ID == "4445026458620"') # (4445028703132, 4445028703157)')
    #
    #     print(ids)
    #
    #     # filter1 = score_df["MERCHANT_ID"].isin([ids])
    #     # filter1 = score_df["MERCHANT_ID"].isin(['4445028703132', '4445028703157'])
    #     filter1 = score_df["MERCHANT_ID"].isin(['4445090630228', '4445028703157'])
    #     score_df = score_df[filter1]
    #
    #     st.write(score_df.astype(str))
    #
    # with col4:
    #     # st.write('')
    #     st.markdown(f'### Results')
    #     st.write('')
    #
    #     for x in df.itertuples():
    #         # st.write(f"Weighted trend for Merchant ID **{x.MERCHANT_ID}**: {x.TREND_WEIGHTED}")
    #         st.write(f"Weighted trend for Merchant **{x.MERCHANT_NAME}** ({x.MERCHANT_ID}): {x.TREND_WEIGHTED}")
    #         display_score(x.TREND_WEIGHTED)

    st.write(df.astype(str))


def display_score(val):
    if val < 0:
        image = Image.open('images/Low.png')
    elif val < 50:
        image = Image.open('images/Medium.png')
    elif val >= 50:
        image = Image.open('images/High.png')
    st.image(image, width=250)
