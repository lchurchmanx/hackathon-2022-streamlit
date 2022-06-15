import streamlit as st
import pandas as pd
import seaborn as sns

from PIL import Image
from matplotlib import pyplot as plt

from app_variables import *


def loan_scoring_all_data(load_func, load_parquet):

    st.markdown('# Loan Score')

    dataset = load_func(path_scoring_dataset)

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
            st.write(f"Merchant **{x.MERCHANT_NAME}** ({x.MERCHANT_ID})")
            display_score(x.SCORE)

            st.write(f"Recommendation: **{recommend_approval(loan[0], x.SCORE)}**")

    st.markdown('# Score Influences')
    merchant_data = load_parquet(path_merge_data_dataset)
    for x in df.itertuples():
        # pass
        score_influences(merchant_data, x.MERCHANT_NAME, x.MERCHANT_ID)
        st.markdown('---')
    # st.write(df.astype(str))


def loan_scoring_limited(load_csv, load_parquet):

    st.markdown('# Loan Score')

    dataset = load_csv(path_scoring_dataset)
    print(dataset.dtypes)
    filter1 = dataset["MERCHANT_ID"].isin(MERCHANT_SUBSET)
    dataset = dataset[filter1]

    col1, col2, col3, col4, col5 = st.columns((.1, 1, .1, 1, .1))

    with col2:

        merchant = [
            st.selectbox('merchant',
                         [x for x in sorted(dataset.MERCHANT_NAME.unique())],
                         key='merchant')
        ]

        loan = [
            st.selectbox('loan',
                         [LOAN_SMALL, LOAN_MEDIUM, LOAN_LARGE],
                         key='loan')
        ]

        df = dataset.query(f'MERCHANT_NAME == "{merchant[0]}"')

    with col4:
        # st.write('')
        # st.markdown(f'### Results')
        st.write('')

        # for x in df.itertuples():
        record = df.iloc[0]
        merchant_name = record['MERCHANT_NAME']
        merchant_id = record['MERCHANT_ID']
        score = record['SCORE']
        st.write(f"Merchant **{merchant_name}** ({merchant_id})")
        display_score(score)

        # st.write(f"Recommendation: **{recommend_approval(loan[0], x.SCORE)}**")

    st.markdown('# Score Influences')
    merchant_data = load_parquet(path_merge_data_dataset)
    score_influences(merchant_data, merchant_name, merchant_id)


def score_influences(merchant_data, merchant_name, merchant_id):
    st.markdown(f'**{merchant_name}** ({merchant_id})')

    merchant_data = merchant_data.query(f'MERCHANT_ID == "{merchant_id}"')
    record = merchant_data.iloc[0]
    print(record)

    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.markdown('### Total sales over 4 years')
        # just hardcode the dates
        new_df = pd.DataFrame({'merchant_id': [record['MERCHANT_ID'], record['MERCHANT_ID'], record['MERCHANT_ID'], record['MERCHANT_ID']],
                               'year': ['2018', '2019', '2020', '2021'],
                               'total_sales': [record['TOT_SALES2018'], record['TOT_SALES2019'], record['TOT_SALES2020'], record['TOT_SALES2021']]})
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=new_df, x="year", y="total_sales")
        st.pyplot(fig)
    with r1_col2:
        st.markdown('### Total CNT over 4 years')
        # just hardcode the dates
        new_df = pd.DataFrame({'merchant_id': [record['MERCHANT_ID'], record['MERCHANT_ID'], record['MERCHANT_ID'], record['MERCHANT_ID']],
                               'year': ['2018', '2019', '2020', '2021'],
                               'total_cnt': [record['TOT_SALE_CNT2018'], record['TOT_SALES_CNT2019'], record['TOT_SALE_CNT2020'], record['TOT_SALE_CNT2021']]})
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=new_df, x="year", y="total_cnt")
        st.pyplot(fig)

    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.markdown('### Population')
        # just hardcode the dates
        new_df = pd.DataFrame({'merchant_id': [record['MERCHANT_ID'], record['MERCHANT_ID'], record['MERCHANT_ID']],
                               'year': ['2018', '2019', '2020'],
                               'total_sales': [record['population_male_2018'] + record['population_female_2018'],
                                               record['population_male_2019'] + record['population_female_2019'],
                                               record['population_male_2020'] + record['population_female_2020']]})
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=new_df, x="year", y="total_sales")
        st.pyplot(fig)
    with r2_col2:
        pass


def display_score(val):
    if val < 1.01:
        image = Image.open('images/Low.png')
    elif val < 1.02:
        image = Image.open('images/Medium.png')
    elif val >= 1.02:
        image = Image.open('images/High.png')
    st.image(image, width=250)


def recommend_approval(loan, score):
    if loan == LOAN_SMALL:
        return "Approved"
    if loan == LOAN_MEDIUM and score > 1.01:
        return "Approved"
    if loan == LOAN_LARGE and score >= 1.02:
        return "Approved"
    return "Denied"
