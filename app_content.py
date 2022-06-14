import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from app_variables import *

LOAN_SMALL = 'Less than 20K'
LOAN_MEDIUM = '20- 50K'
LOAN_LARGE = '50K-120K'


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

    dataset = load_csv(path_scoring_dataset)

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
                         [LOAN_SMALL, LOAN_MEDIUM, LOAN_LARGE],
                         key='loan')
        ]

        df = df.query(f'MERCHANT_NAME == "{merchant[0]}"')

    with col4:
        # st.write('')
        st.markdown(f'### Results')
        st.write('')

        for x in df.itertuples():
            st.write(f"Weighted trend for Merchant **{x.MERCHANT_NAME}** ({x.MERCHANT_ID}): {x.SCORE}")
            display_score(x.SCORE)

            st.write(f"Recommendation: {recommend_approval(loan[0], x.SCORE)}")

    # st.write(df.astype(str))


def display_score(val):
    if val < 1:
        image = Image.open('images/Low.png')
    elif val < 2:
        image = Image.open('images/Medium.png')
    elif val >= 2:
        image = Image.open('images/High.png')
    st.image(image, width=250)


def recommend_approval(loan, score):
    print(loan, score)
    if loan == LOAN_SMALL:
        return "Approve loan"
    if loan == LOAN_MEDIUM and score > 1:
        return "Approve loan"
    if loan == LOAN_LARGE and score >=2:
        return "Approve loan"
    return "Do not approve loan"

