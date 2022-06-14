import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from loan_scoring import *
import plotly.graph_objects as go
import plotly.express as px



def load_csv(path):
    dataset = pd.read_csv(os.path.join(path))
    return dataset


def load_parquet(path):
    return pd.read_parquet(path)


def set_home():
    st.write(intro, unsafe_allow_html=True)
    st.write(intro_context, unsafe_allow_html=True)


def set_data_quality():
    st.markdown('# Data Quality')
    st.markdown('### Quick glance on all the source data and review data quality')
    st.markdown('#### Merchant Data')
    dataset = load_csv(path_merchant)

    numerical = dataset.select_dtypes(include=['number', 'bool', 'datetime64[ns]', 'timedelta64'])
    st.table(pd.DataFrame([[dataset.shape[0], dataset.shape[1],
                            (dataset.shape[1] - numerical.shape[1]), numerical.shape[1]]],
                          columns=["Row Count", "Column Count", "Nominal Column Count",
                                   "Numeric Column Count"], index=[""]))
    useless = dataset[dataset.isnull().sum(axis=1) > (dataset.shape[1] / 2)]
    uselessRows_count = useless.shape[0]
    if uselessRows_count > 0:
        st.write(str(uselessRows_count), "rows may be useless:", useless)
        st.write("")

    duplicated = dataset[dataset.duplicated()]
    duplicatedRows_count = duplicated.shape[0]
    if duplicatedRows_count == 0:
        st.success("There is no duplicated rows in the dataset.")
    else:
        st.write("There are", str(duplicatedRows_count), "duplicated rows in the dataset:",
                 dataset[dataset.duplicated()])
    st.write("---")
    st.markdown('#### Banking Data')

    dataset = load_csv(path_banking)

    numerical = dataset.select_dtypes(include=['number', 'bool', 'datetime64[ns]', 'timedelta64'])
    st.table(pd.DataFrame([[dataset.shape[0], dataset.shape[1],
                            (dataset.shape[1] - numerical.shape[1]), numerical.shape[1]]],
                          columns=["Row Count", "Column Count", "Nominal Column Count",
                                   "Numeric Column Count"], index=[""]))
    useless = dataset[dataset.isnull().sum(axis=1) > (dataset.shape[1] / 2)]
    uselessRows_count = useless.shape[0]
    if uselessRows_count > 0:
        st.write(str(uselessRows_count), "rows may be useless:", useless)
        st.write("")

    duplicated = dataset[dataset.duplicated()]
    duplicatedRows_count = duplicated.shape[0]
    if duplicatedRows_count == 0:
        st.success("There is no duplicated rows in the dataset.")
    else:
        st.write("There are", str(duplicatedRows_count), "duplicated rows in the dataset:",
                 dataset[dataset.duplicated()])

    st.write("---")
    st.markdown('#### AGG Data')
    dataset = load_csv(path_agg)

    numerical = dataset.select_dtypes(include=['number', 'bool', 'datetime64[ns]', 'timedelta64'])
    st.table(pd.DataFrame([[dataset.shape[0], dataset.shape[1],
                            (dataset.shape[1] - numerical.shape[1]), numerical.shape[1]]],
                          columns=["Row Count", "Column Count", "Nominal Column Count",
                                   "Numeric Column Count"], index=[""]))
    useless = dataset[dataset.isnull().sum(axis=1) > (dataset.shape[1] / 2)]
    uselessRows_count = useless.shape[0]
    if uselessRows_count > 0:
        st.write(str(uselessRows_count), "rows may be useless:", useless)
        st.write("")

    duplicated = dataset[dataset.duplicated()]
    duplicatedRows_count = duplicated.shape[0]
    if duplicatedRows_count == 0:
        st.success("There is no duplicated rows in the dataset.")
    else:
        st.write("There are", str(duplicatedRows_count), "duplicated rows in the dataset:",
                 dataset[dataset.duplicated()])


def set_feature_importance():
    st.title('Feature Importance Review')

    st.markdown('## DataFrame `Feature Importance`')
    st.markdown(
        '#### Utilized machine learning algorithm (Random Forest) and combined Census Data to determine feature importance.')
    ds = load_csv(path_feature_importance)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### Review Feature Importance with a bar chart against target column `TOT_SALES2021`')
    df_bar_commodity = ds.groupby('name')['score'].agg(['mean']).reset_index()
    df_bar_commodity.rename(columns={'mean': 'feature_importance'}, inplace=True)
    fig = px.bar(df_bar_commodity.sort_values(by='feature_importance', ascending=False).iloc[:10],
                 x="feature_importance",
                 y="name",
                 color="name")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def set_data_exploration():
    global selected_column_country_year, selected_column_commodity_year
    dataset = load_parquet(path_merge_data_dataset)

    st.markdown('# Basic Statistics')
    st.markdown('### Review descriptive statistics for `Combined Dataset`')
    cols = ["ZIPCODE", "MCC", "BILLING_STATE", "CITY", "MERCHANT_ID"]
    dataset.drop(cols, axis=1, inplace=True)
    # st.write(dataset.columns)
    st.write(dataset.describe())

    # st.markdown('------')
    #
    # st.markdown('### Review descriptive statistics for `loss_percentage` per `category value` selected')
    # row3_space1, row3_1, row3_space2, row3_2, row3_space3, row3_3, row3_space4 = st.columns((.1, 1, .1, 1, .1, 1, .1))
    # with row3_1:
    #     select_column_country = [
    #         st.selectbox('',
    #                      dataset.country.unique(),
    #                      key='select_column_country')]
    #
    #     for selection in select_column_country:
    #         desc_dataset = dataset[['country', 'loss_percentage']].query(f'country == "{selection}"')
    #         st.write(desc_dataset[['loss_percentage']].describe())
    # with row3_2:
    #     select_column_year = [
    #         st.selectbox('',
    #                      dataset.year.unique(),
    #                      key='select_column_year')]
    #     for selection in select_column_year:
    #         desc_dataset = dataset[['year', 'loss_percentage']].query(f'year == {selection}')
    #         st.write(desc_dataset[['loss_percentage']].describe())
    # with row3_3:
    #     select_column_commodity = [
    #         st.selectbox('',
    #                      dataset.commodity.unique(),
    #                      key='select_column_commodity')]
    #     for selection in select_column_commodity:
    #         desc_dataset = dataset[['commodity', 'loss_percentage']].query(f'commodity == "{selection}"')
    #         st.write(desc_dataset[['loss_percentage']].describe())
    #
    # st.markdown('------')
    #
    # st.markdown('### Review yearly loss percentage changes `line chart`')
    #
    # row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((.1, 1, .1, 1, .1))
    #
    # with row4_1:
    #     select_column_country_year = [
    #         st.selectbox('',
    #                      dataset.country.unique(),
    #                      key='select_column_country_year')]
    #     for selection in select_column_country_year:
    #         selected_column_country_year = selection
    #
    #     commodity_unique_value = dataset.query(f'country == "{selected_column_country_year}"')
    #     commodity_unique_value = commodity_unique_value.commodity.unique()
    #
    #     select_column_commodity_year = [
    #         st.selectbox('',
    #                      commodity_unique_value,
    #                      key='select_column_commodity_year')]
    #
    #     for selection in select_column_commodity_year:
    #         selected_column_commodity_year = selection
    #
    #     dataset_groomed = dataset[['country', 'commodity', 'year', 'loss_percentage']].query(
    #         f'country == "{selected_column_country_year}" and commodity == "{selected_column_commodity_year}"')
    #     fig = plt.figure(figsize=(10, 4))
    #     # plt.xticks(rotation=90)
    #     sns.lineplot(data=dataset_groomed, x="year", y="loss_percentage")
    #     st.pyplot(fig)
    # with row4_2:
    #     select_column_country_year = [
    #         st.selectbox('',
    #                      dataset.country.unique(),
    #                      key='select_column_country_year_2')]
    #     for selection in select_column_country_year:
    #         selected_column_country_year = selection
    #
    #     food_supply_stage_unique_value = dataset.query(f'country == "{selected_column_country_year}"')
    #     food_supply_stage_unique_value = food_supply_stage_unique_value.food_supply_stage.unique()
    #
    #     select_column_food_supply_stage_year_2 = [
    #         st.selectbox('',
    #                      food_supply_stage_unique_value,
    #                      key='select_column_food_supply_stage_year_2')]
    #
    #     for selection in select_column_food_supply_stage_year_2:
    #         selected_column_food_supply_stage_year_2 = selection
    #
    #     dataset_groomed_2 = dataset[['country', 'food_supply_stage', 'year', 'loss_percentage']].query(
    #         f'country == "{selected_column_country_year}" and food_supply_stage == "{selected_column_food_supply_stage_year_2}"')
    #     fig = plt.figure(figsize=(10, 4))
    #     # plt.xticks(rotation=90)
    #     sns.lineplot(data=dataset_groomed_2, x="year", y="loss_percentage")
    #     st.pyplot(fig)
    #
    # st.markdown('------')
    #
    # st.markdown('### Review yearly loss percentage changes `scatter plot`')
    #
    # row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns((.1, 1, .1, 1, .1))
    #
    # with row5_1:
    #     select_column_country_year_scatter = [
    #         st.selectbox('',
    #                      dataset.country.unique(),
    #                      key='select_column_country_year_scatter')]
    #     for selection in select_column_country_year_scatter:
    #         selected_column_country_year_scatter = selection
    #
    #     commodity_unique_value_scatter = dataset.query(f'country == "{selected_column_country_year_scatter}"')
    #     commodity_unique_value_scatter = commodity_unique_value_scatter.commodity.unique()
    #
    #     select_column_commodity_year_scatter = [
    #         st.selectbox('',
    #                      commodity_unique_value_scatter,
    #                      key='select_column_commodity_year_scatter')]
    #
    #     for selection in select_column_commodity_year_scatter:
    #         selected_column_commodity_year_scatter = selection
    #
    #     dataset_groomed_scatter = dataset[['country', 'commodity', 'year', 'loss_percentage']].query(
    #         f'country == "{selected_column_country_year_scatter}" and commodity == "{selected_column_commodity_year_scatter}"')
    #     fig = plt.figure(figsize=(10, 4))
    #     # plt.xticks(rotation=90)
    #     sns.scatterplot(data=dataset_groomed_scatter, x="year", y="loss_percentage")
    #     st.pyplot(fig)
    #
    # with row5_2:
    #     select_column_country_year_scatter = [
    #         st.selectbox('',
    #                      dataset.country.unique(),
    #                      key='select_column_country_year_2_scatter')]
    #     for selection in select_column_country_year_scatter:
    #         selected_column_country_year_scatter = selection
    #
    #     food_supply_stage_unique_value_scatter = dataset.query(f'country == "{selected_column_country_year}"')
    #     food_supply_stage_unique_value_scatter = food_supply_stage_unique_value_scatter.food_supply_stage.unique()
    #
    #     select_column_food_supply_stage_year_2_scatter = [
    #         st.selectbox('',
    #                      food_supply_stage_unique_value_scatter,
    #                      key='select_column_food_supply_stage_year_2_scatter')]
    #
    #     for selection in select_column_food_supply_stage_year_2_scatter:
    #         selected_column_food_supply_stage_year_2_scatter = selection
    #
    #     dataset_groomed_2_scatter = dataset[['country', 'food_supply_stage', 'year', 'loss_percentage']].query(
    #         f'country == "{selected_column_country_year_scatter}" and food_supply_stage == "{selected_column_food_supply_stage_year_2_scatter}"')
    #     fig = plt.figure(figsize=(10, 4))
    #     # plt.xticks(rotation=90)
    #     sns.scatterplot(data=dataset_groomed_2_scatter, x="year", y="loss_percentage")
    #     st.pyplot(fig)


def preview_data():
    st.title('Data Preview')

    st.markdown('### DataFrame `Scoring`')
    ds = load_csv(path_scoring_dataset).head(200)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Ranking`')
    ds = load_parquet(path_ranking_dataset).head(200)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Merchant Modeling`')
    ds = load_parquet(path_modeling_merchant_dataset).head(200)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Merged Data`')
    ds = load_parquet(path_merge_data_dataset).head(200)
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))

    st.markdown('### DataFrame `Bank Data`')
    st.markdown(
        '')
    dataset = load_csv(path_bank_dataset).head(200)
    st.markdown(f'{len(dataset.index)} entries  |  {len(dataset.columns)} columns')
    st.write(dataset.astype(str))

    ds = load_csv(path_merchant_dataset).head(200)
    st.markdown('### DataFrame `Merchant Data`')
    st.markdown(
        '')
    st.markdown(f'{len(ds.index)} entries  |  {len(ds.columns)} columns')
    st.write(ds.astype(str))


def in_progress():
    st.markdown('# In Progress...')


def loan_scoring():
    loan_scoring_limited(load_csv)  # only choose between a subset of merchants
    # loan_scoring_all_data(load_csv)  # this is the full form to choose state/city/merchant

