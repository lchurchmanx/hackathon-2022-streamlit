import streamlit as st
from PIL import Image

from app_variables import *


def loan_scoring_all_data(load_func):

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

    # st.write(df.astype(str))


def loan_scoring_limited(load_csv):

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

    score_influences()

    # st.write(df.astype(str))

def score_influences():
    st.markdown('# Score Influences')
    st.markdown('')


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
