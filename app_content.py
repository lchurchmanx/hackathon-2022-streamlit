import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from iso3166 import countries
import plotly.graph_objects as go
from app_variables import *
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from st_btn_select import st_btn_select
import operator
import category_encoders as ce
from sklearn.model_selection import train_test_split


def rename(country):
    try:
        return countries.get(country).alpha3
    except:
        return np.nan


def load_csv():
    dataset = pd.read_csv(os.path.join(path_dataset))
    return dataset


def set_home():
    st.write(intro, unsafe_allow_html=True)
    st.write(intro_context, unsafe_allow_html=True)


def set_data():
    dataset = load_csv()

    st.title('Data Preview')
    # st.write(dataset_intro, unsafe_allow_html=True)

    st.markdown('### DataFrame `Food Loss and Waste Database`')
    st.markdown(
        '')
    st.markdown(f'{len(dataset.index)} entries  |  {len(dataset.columns)} columns')
    st.write(dataset.astype(str))

    st.markdown(f'### There are {len(dataset.columns)} columns that include: ')
    st.markdown(column_table, unsafe_allow_html=True)


def set_data_quality():
    dataset = load_csv()

    st.markdown('# Data Quality')

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


def set_data_exploration():
    global selected_column_country_year, selected_column_commodity_year
    dataset = load_csv()

    st.markdown('# Basic Statistics')
    st.markdown('### Review descriptive statistics for `loss_percentage`')
    st.write(dataset['loss_percentage'].describe())

    st.markdown('------')

    st.markdown('### Review descriptive statistics for `loss_percentage` per `category value` selected')
    row3_space1, row3_1, row3_space2, row3_2, row3_space3, row3_3, row3_space4 = st.columns((.1, 1, .1, 1, .1, 1, .1))
    with row3_1:
        select_column_country = [
            st.selectbox('',
                         dataset.country.unique(),
                         key='select_column_country')]

        for selection in select_column_country:
            desc_dataset = dataset[['country', 'loss_percentage']].query(f'country == "{selection}"')
            st.write(desc_dataset[['loss_percentage']].describe())
    with row3_2:
        select_column_year = [
            st.selectbox('',
                         dataset.year.unique(),
                         key='select_column_year')]
        for selection in select_column_year:
            desc_dataset = dataset[['year', 'loss_percentage']].query(f'year == {selection}')
            st.write(desc_dataset[['loss_percentage']].describe())
    with row3_3:
        select_column_commodity = [
            st.selectbox('',
                         dataset.commodity.unique(),
                         key='select_column_commodity')]
        for selection in select_column_commodity:
            desc_dataset = dataset[['commodity', 'loss_percentage']].query(f'commodity == "{selection}"')
            st.write(desc_dataset[['loss_percentage']].describe())

    st.markdown('------')

    st.markdown('### Review yearly loss percentage changes `line chart`')

    row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((.1, 1, .1, 1, .1))

    with row4_1:
        select_column_country_year = [
            st.selectbox('',
                         dataset.country.unique(),
                         key='select_column_country_year')]
        for selection in select_column_country_year:
            selected_column_country_year = selection

        commodity_unique_value = dataset.query(f'country == "{selected_column_country_year}"')
        commodity_unique_value = commodity_unique_value.commodity.unique()

        select_column_commodity_year = [
            st.selectbox('',
                         commodity_unique_value,
                         key='select_column_commodity_year')]

        for selection in select_column_commodity_year:
            selected_column_commodity_year = selection

        dataset_groomed = dataset[['country', 'commodity', 'year', 'loss_percentage']].query(
            f'country == "{selected_column_country_year}" and commodity == "{selected_column_commodity_year}"')
        fig = plt.figure(figsize=(10, 4))
        # plt.xticks(rotation=90)
        sns.lineplot(data=dataset_groomed, x="year", y="loss_percentage")
        st.pyplot(fig)
    with row4_2:
        select_column_country_year = [
            st.selectbox('',
                         dataset.country.unique(),
                         key='select_column_country_year_2')]
        for selection in select_column_country_year:
            selected_column_country_year = selection

        food_supply_stage_unique_value = dataset.query(f'country == "{selected_column_country_year}"')
        food_supply_stage_unique_value = food_supply_stage_unique_value.food_supply_stage.unique()

        select_column_food_supply_stage_year_2 = [
            st.selectbox('',
                         food_supply_stage_unique_value,
                         key='select_column_food_supply_stage_year_2')]

        for selection in select_column_food_supply_stage_year_2:
            selected_column_food_supply_stage_year_2 = selection

        dataset_groomed_2 = dataset[['country', 'food_supply_stage', 'year', 'loss_percentage']].query(
            f'country == "{selected_column_country_year}" and food_supply_stage == "{selected_column_food_supply_stage_year_2}"')
        fig = plt.figure(figsize=(10, 4))
        # plt.xticks(rotation=90)
        sns.lineplot(data=dataset_groomed_2, x="year", y="loss_percentage")
        st.pyplot(fig)

    st.markdown('------')

    st.markdown('### Review yearly loss percentage changes `scatter plot`')

    row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns((.1, 1, .1, 1, .1))

    with row5_1:
        select_column_country_year_scatter = [
            st.selectbox('',
                         dataset.country.unique(),
                         key='select_column_country_year_scatter')]
        for selection in select_column_country_year_scatter:
            selected_column_country_year_scatter = selection

        commodity_unique_value_scatter = dataset.query(f'country == "{selected_column_country_year_scatter}"')
        commodity_unique_value_scatter = commodity_unique_value_scatter.commodity.unique()

        select_column_commodity_year_scatter = [
            st.selectbox('',
                         commodity_unique_value_scatter,
                         key='select_column_commodity_year_scatter')]

        for selection in select_column_commodity_year_scatter:
            selected_column_commodity_year_scatter = selection

        dataset_groomed_scatter = dataset[['country', 'commodity', 'year', 'loss_percentage']].query(
            f'country == "{selected_column_country_year_scatter}" and commodity == "{selected_column_commodity_year_scatter}"')
        fig = plt.figure(figsize=(10, 4))
        # plt.xticks(rotation=90)
        sns.scatterplot(data=dataset_groomed_scatter, x="year", y="loss_percentage")
        st.pyplot(fig)

    with row5_2:
        select_column_country_year_scatter = [
            st.selectbox('',
                         dataset.country.unique(),
                         key='select_column_country_year_2_scatter')]
        for selection in select_column_country_year_scatter:
            selected_column_country_year_scatter = selection

        food_supply_stage_unique_value_scatter = dataset.query(f'country == "{selected_column_country_year}"')
        food_supply_stage_unique_value_scatter = food_supply_stage_unique_value_scatter.food_supply_stage.unique()

        select_column_food_supply_stage_year_2_scatter = [
            st.selectbox('',
                         food_supply_stage_unique_value_scatter,
                         key='select_column_food_supply_stage_year_2_scatter')]

        for selection in select_column_food_supply_stage_year_2_scatter:
            selected_column_food_supply_stage_year_2_scatter = selection

        dataset_groomed_2_scatter = dataset[['country', 'food_supply_stage', 'year', 'loss_percentage']].query(
            f'country == "{selected_column_country_year_scatter}" and food_supply_stage == "{selected_column_food_supply_stage_year_2_scatter}"')
        fig = plt.figure(figsize=(10, 4))
        # plt.xticks(rotation=90)
        sns.scatterplot(data=dataset_groomed_2_scatter, x="year", y="loss_percentage")
        st.pyplot(fig)


def set_data_overview():
    dataset = load_csv()

    st.markdown('# Data Overview')

    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((.1, 1, .1, 1, .1))

    with row1_1:
        st.markdown('### Average loss percentage for Canada')

        fig = go.Figure(go.Indicator(
            mode="number",
            value=round(float(dataset.query('country == "Canada"')['loss_percentage'].agg(['mean'])), 2),
            number={'suffix': "%"}))
        st.plotly_chart(fig, use_container_width=True)

    with row1_2:
        st.markdown('### Average loss percentage for all countries')

        fig = go.Figure(go.Indicator(
            mode="number",
            value=round(float(dataset['loss_percentage'].agg(['mean'])), 2),
            number={'suffix': "%"}))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('------')

    row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns((.1, 1, .1, 1, .1))

    with row6_1:
        st.markdown('### Review average loss percentage `bar chart` by `food supply stage` (Canada)')

        df_bar_food_supply_stage_Canada = dataset.query('country == "Canada"').groupby('food_supply_stage')[
            'loss_percentage'].agg(['mean']).reset_index()
        df_bar_food_supply_stage_Canada.rename(columns={'mean': 'avg_loss_percentage'}, inplace=True)
        fig = px.bar(df_bar_food_supply_stage_Canada.sort_values(by='avg_loss_percentage', ascending=False),
                     x="avg_loss_percentage", y="food_supply_stage",
                     color="food_supply_stage")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with row6_2:
        st.markdown('### Review average loss percentage `bar chart` by `food supply stage` (all countries)')
        df_bar_food_supply_stage = dataset.groupby('food_supply_stage')['loss_percentage'].agg(['mean']).reset_index()
        df_bar_food_supply_stage.rename(columns={'mean': 'avg_loss_percentage'}, inplace=True)
        fig = px.bar(df_bar_food_supply_stage.sort_values(by='avg_loss_percentage', ascending=False),
                     x="avg_loss_percentage", y="food_supply_stage",
                     color="food_supply_stage")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('------')

    row7_space1, row7_1, row7_space2, row7_2, row7_space3 = st.columns((.1, 1, .1, 1, .1))

    with row7_1:
        st.markdown('### Review average loss percentage `bar chart` by `commodity` (Canada)')

        df_bar_commodity_Canada = dataset.query('country == "Canada"').groupby('commodity')[
            'loss_percentage'].agg(['mean']).reset_index()
        df_bar_commodity_Canada.rename(columns={'mean': 'avg_loss_percentage'}, inplace=True)
        fig = px.bar(df_bar_commodity_Canada.sort_values(by='avg_loss_percentage', ascending=False).iloc[:10],
                     x="avg_loss_percentage", y="commodity",
                     color="commodity")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with row7_2:
        st.markdown('### Review average loss percentage `bar chart` by `commodity` (all countries)')
        df_bar_commodity = dataset.groupby('commodity')['loss_percentage'].agg(['mean']).reset_index()
        df_bar_commodity.rename(columns={'mean': 'avg_loss_percentage'}, inplace=True)
        fig = px.bar(df_bar_commodity.sort_values(by='avg_loss_percentage', ascending=False).iloc[:10],
                     x="avg_loss_percentage",
                     y="commodity",
                     color="commodity")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('------')

    row8_space1, row8_1, row8_space2, row8_2, row8_space3 = st.columns((.1, 1, .1, 1, .1))

    with row8_1:
        st.markdown('### Review average loss percentage `bar chart` by `treatment` (Canada)')

        df_bar_treatment_Canada = dataset.query('country == "Canada"').groupby('treatment')[
            'loss_percentage'].agg(['mean']).reset_index()
        df_bar_treatment_Canada.rename(columns={'mean': 'avg_loss_percentage'}, inplace=True)
        fig = px.bar(df_bar_treatment_Canada.sort_values(by='avg_loss_percentage', ascending=False).iloc[:10],
                     x="avg_loss_percentage", y="treatment",
                     color="treatment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with row8_2:
        st.markdown('### Review average loss percentage `bar chart` by `treatment` (all countries)')
        df_bar_treatment = dataset.groupby('treatment')['loss_percentage'].agg(['mean']).reset_index()
        df_bar_treatment.rename(columns={'mean': 'avg_loss_percentage'}, inplace=True)
        fig = px.bar(df_bar_treatment.sort_values(by='avg_loss_percentage', ascending=False).iloc[:10],
                     x="avg_loss_percentage",
                     y="treatment",
                     color="treatment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def set_map_plots():
    global selected_column_food_supply_stage_map, selected_column_commodity_map
    dataset = load_csv()
    dataset['country_alias'] = dataset['country'].apply(rename)

    st.markdown('# Map Plots')
    st.markdown('### Loss percentage per `commodity`')
    df = dataset[['country_alias', 'country', 'commodity', 'year', 'loss_percentage']].dropna()
    df = df[df['year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]

    commodity_unique_value_map = df.commodity.unique()

    select_column_commodity_map = [
        st.selectbox('',
                     commodity_unique_value_map,
                     key='select_column_commodity_map')]

    for selection in select_column_commodity_map:
        selected_column_commodity_map = selection

    fig = px.scatter_geo(df.query(
        f'commodity == "{selected_column_commodity_map}"').sort_values('year', ascending=True),
                         locations="country_alias", size='loss_percentage',
                         hover_name="country", color='country', animation_frame="year", animation_group="country")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('------')

    st.markdown('### Loss percentage per `food supply stage`')

    df = dataset[['country_alias', 'country', 'food_supply_stage', 'year', 'loss_percentage']].dropna()
    df = df[df['year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]

    food_supply_stage_unique_value_map = df.food_supply_stage.unique()

    select_column_food_supply_stage_map = [
        st.selectbox('',
                     food_supply_stage_unique_value_map,
                     key='select_column_food_supply_stage_map')]

    for selection in select_column_food_supply_stage_map:
        selected_column_food_supply_stage_map = selection

    fig = px.scatter_geo(df.query(
        f'food_supply_stage == "{selected_column_food_supply_stage_map}"').sort_values('year', ascending=True),
                         locations="country_alias", size='loss_percentage',
                         hover_name="country", color='country', animation_frame="year", animation_group="country")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('------')

    st.markdown('### Average loss percentage per `commodity`')

    # prepare avg loss percentage

    food_waste_df_commodity_commodity = dataset[['country', 'commodity', 'year', 'loss_percentage']]
    food_waste_df_commodity = food_waste_df_commodity_commodity.dropna()

    food_waste_df_pivot_commodity = pd.pivot_table(food_waste_df_commodity, values='loss_percentage',
                                                   index=['country', 'commodity'],
                                                   columns=['year'],
                                                   aggfunc=np.mean)

    food_waste_df_pivot_commodity = food_waste_df_pivot_commodity.dropna(axis=1, how='all')
    food_waste_df_pivot_commodity = food_waste_df_pivot_commodity.reset_index()
    food_waste_df_pivot_commodity = food_waste_df_pivot_commodity.melt(id_vars=["country", "commodity"],
                                                                       var_name="year",
                                                                       value_name="avg_loss_percentage")
    food_waste_df_pivot_commodity = food_waste_df_pivot_commodity.dropna()
    food_waste_df_pivot_commodity["avg_loss_percentage"] = pd.to_numeric(
        food_waste_df_pivot_commodity["avg_loss_percentage"])
    food_waste_df_pivot_commodity = food_waste_df_pivot_commodity[
        food_waste_df_pivot_commodity['year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]
    food_waste_df_pivot_commodity['country_alias'] = food_waste_df_pivot_commodity['country'].apply(rename)

    commodity_unique_value_map_avg = food_waste_df_pivot_commodity.commodity.unique()

    select_column_commodity_map_avg = [
        st.selectbox('',
                     commodity_unique_value_map_avg,
                     key='select_column_commodity_map_avg')]

    for selection in select_column_commodity_map_avg:
        selected_column_commodity_map_avg = selection

    fig = px.scatter_geo(food_waste_df_pivot_commodity.query(
        f'commodity == "{selected_column_commodity_map_avg}"').sort_values('year', ascending=True),
                         locations="country_alias", size='avg_loss_percentage',
                         hover_name="country", color='country', animation_frame="year", animation_group="country")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('------')

    st.markdown('### Average loss percentage per `food supply stage`')

    # prepare avg loss percentage

    food_waste_df_fss = dataset[['country', 'food_supply_stage', 'year', 'loss_percentage']]
    food_waste_df_fss = food_waste_df_fss.dropna()

    food_waste_df_pivot_fss = pd.pivot_table(food_waste_df_fss, values='loss_percentage',
                                             index=['country', 'food_supply_stage'],
                                             columns=['year'],
                                             aggfunc=np.mean)

    food_waste_df_pivot_fss = food_waste_df_pivot_fss.dropna(axis=1, how='all')
    food_waste_df_pivot_fss = food_waste_df_pivot_fss.reset_index()
    food_waste_df_pivot_fss = food_waste_df_pivot_fss.melt(id_vars=["country", "food_supply_stage"],
                                                           var_name="year",
                                                           value_name="avg_loss_percentage")
    food_waste_df_pivot_fss = food_waste_df_pivot_fss.dropna()
    food_waste_df_pivot_fss["avg_loss_percentage"] = pd.to_numeric(food_waste_df_pivot_fss["avg_loss_percentage"])
    food_waste_df_pivot_fss = food_waste_df_pivot_fss[
        food_waste_df_pivot_fss['year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]
    food_waste_df_pivot_fss['country_alias'] = food_waste_df_pivot_fss['country'].apply(rename)

    food_supply_stage_unique_value_map_avg = food_waste_df_pivot_fss.food_supply_stage.unique()

    select_column_food_supply_stage_map_avg = [
        st.selectbox('',
                     food_supply_stage_unique_value_map_avg,
                     key='select_column_food_supply_stage_map_avg')]

    for selection in select_column_food_supply_stage_map_avg:
        selected_column_food_supply_stage_map_avg = selection

    fig = px.scatter_geo(food_waste_df_pivot_fss.query(
        f'food_supply_stage == "{selected_column_food_supply_stage_map_avg}"').sort_values('year', ascending=True),
                         locations="country_alias", size='avg_loss_percentage',
                         hover_name="country", color='country', animation_frame="year", animation_group="country")
    st.plotly_chart(fig, use_container_width=True)


def set_relationship():
    food_waste_df = pd.read_csv(path_dataset)
    gdp_df = pd.read_csv(path_gdp)
    gdppc_df = pd.read_csv(path_gdp_per_capita)
    food_waste_df = food_waste_df[['country', 'year', 'loss_percentage']]
    food_waste_df = food_waste_df.dropna()

    food_waste_df_pivot = pd.pivot_table(food_waste_df, values='loss_percentage',
                                         index=['country'],
                                         columns=['year'],
                                         aggfunc=np.mean)

    food_waste_df_pivot = food_waste_df_pivot.dropna(axis=1, how='all')
    food_waste_df_pivot = food_waste_df_pivot.reset_index()

    gdp_df.rename(
        columns={'Country Name': 'country', '1990 [YR1990]': '1990', '2000 [YR2000]': '2000', '2011 [YR2011]': '2011',
                 '2012 [YR2012]': '2012', '2013 [YR2013]': '2013', '2014 [YR2014]': '2014', '2015 [YR2015]': '2015',
                 '2016 [YR2016]': '2016', '2017 [YR2017]': '2017', '2018 [YR2018]': '2018', '2019 [YR2019]': '2019',
                 '2020 [YR2020]': '2020'}, inplace=True)
    gdp_df.drop(['Series Name', 'Series Code', 'Country Code'], axis=1, inplace=True)
    gdp_df = gdp_df.melt(id_vars=["country"],
                         var_name="year",
                         value_name="gdp")
    gdp_df['country_code'] = gdp_df['country'].apply(rename)
    gdp_df['lookup_columns'] = gdp_df['country_code'] + gdp_df['year']

    food_waste_df_pivot = food_waste_df_pivot.melt(id_vars=["country"],
                                                   var_name="year",
                                                   value_name="avg_loss_percentage")
    food_waste_df_pivot = food_waste_df_pivot.dropna()
    food_waste_df_pivot['country_code'] = food_waste_df_pivot['country'].apply(rename)
    food_waste_df_pivot['lookup_columns'] = food_waste_df_pivot['country_code'] + food_waste_df_pivot['year'].astype(
        str)

    gdppc_df.drop(['Indicator Name', 'Indicator Code', 'Unnamed: 65', 'Country Code'], axis=1, inplace=True)
    gdppc_df.rename(columns={'Country Name': 'country'}, inplace=True)
    gdppc_df = gdppc_df.melt(id_vars=["country"],
                             var_name="year",
                             value_name="gdp_per_capita")
    gdppc_df['country_code'] = gdppc_df['country'].apply(rename)
    gdppc_df['lookup_columns'] = gdppc_df['country_code'] + gdppc_df['year']
    gdppc_df = gdppc_df.dropna()

    merged_df = food_waste_df_pivot.merge(gdp_df, on='lookup_columns', how='left')
    merged_df = merged_df.merge(gdppc_df, on='lookup_columns', how='left')
    merged_df = merged_df.dropna()
    final_df = merged_df[['country_x', 'year_x', 'avg_loss_percentage', 'gdp', 'gdp_per_capita']]
    final_df.rename(columns={'country_x': 'country', 'year_x': 'year'}, inplace=True)
    final_df["avg_loss_percentage"] = pd.to_numeric(final_df["avg_loss_percentage"])
    final_df = final_df[~final_df['gdp'].isin([".."])]
    final_df = final_df[~final_df['gdp_per_capita'].isin([".."])]
    final_df["gdp"] = pd.to_numeric(final_df["gdp"])
    final_df = final_df[final_df['year'].isin([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])]
    final_df = final_df.round(3)

    st.markdown('# Data Relationship')
    st.markdown('### Relationship between `average loss percentage` and `GDP PER CAPITA`')

    fig = px.scatter(final_df, x="gdp_per_capita", y="avg_loss_percentage", animation_frame="year",
                     hover_name="country", color="country", size="gdp",
                     animation_group="country", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def set_feature_importance():
    # prepare dataframe
    food_waste_df = pd.read_csv(path_dataset)
    gdp_df = pd.read_csv(path_gdp)
    gdppc_df = pd.read_csv(path_gdp_per_capita)

    gdp_df.rename(
        columns={'Country Name': 'country', '1990 [YR1990]': '1990', '2000 [YR2000]': '2000', '2011 [YR2011]': '2011',
                 '2012 [YR2012]': '2012', '2013 [YR2013]': '2013', '2014 [YR2014]': '2014', '2015 [YR2015]': '2015',
                 '2016 [YR2016]': '2016', '2017 [YR2017]': '2017', '2018 [YR2018]': '2018', '2019 [YR2019]': '2019',
                 '2020 [YR2020]': '2020'}, inplace=True)
    gdp_df.drop(['Series Name', 'Series Code', 'Country Code'], axis=1, inplace=True)
    gdp_df = gdp_df.melt(id_vars=["country"],
                         var_name="year",
                         value_name="gdp")
    gdp_df['country_code'] = gdp_df['country'].apply(rename)
    gdp_df['lookup_columns'] = gdp_df['country_code'] + gdp_df['year']
    gdp_df = gdp_df.dropna()

    gdppc_df.drop(['Indicator Name', 'Indicator Code', 'Country Code'], axis=1, inplace=True)
    gdppc_df.rename(columns={'Country Name': 'country'}, inplace=True)
    gdppc_df = gdppc_df.melt(id_vars=["country"],
                             var_name="year",
                             value_name="gdp_per_capita")
    gdppc_df['country_code'] = gdppc_df['country'].apply(rename)
    gdppc_df['lookup_columns'] = gdppc_df['country_code'] + gdppc_df['year']
    gdppc_df = gdppc_df.dropna()

    food_waste_df['country_code'] = food_waste_df['country'].apply(rename)
    food_waste_df['lookup_columns'] = food_waste_df['country_code'] + food_waste_df['year'].astype(str)

    merged_df = food_waste_df.merge(gdp_df, on='lookup_columns', how='left')
    merged_df = merged_df.merge(gdppc_df, on='lookup_columns', how='left')
    final_df = merged_df[~merged_df['gdp'].isin([".."])]
    final_df = final_df[~final_df['gdp_per_capita'].isin([".."])]

    st.markdown('# Feature Importance')
    st.markdown(
        '### Review feature importance among `year`, `GDP`, `GDP per CAPITA`,`commodity`, `food_supply_stage`, `cause_of_loss`, `treatment`')
    st.markdown(
        'This section utilized XGBoost to automatically provide estimates of feature importance from a trained predictive model. '
        'Four features were used with XGBoost to find the feature importance.')

    row9_space1, row9_1, row9_space2, row9_2, row9_space3 = st.columns((.1, 1, .1, 1, .1))

    with row9_1:
        st.markdown('### Canada')
        image = Image.open('feature_importance_Canada.png')
        st.image(image)

    with row9_2:
        st.markdown('### All countries')
        image = Image.open('feature_importance_all.png')
        st.image(image)

    st.markdown('------')

    st.markdown(
        '### Predict loss percentage by entering 7 feature parameters - `year`, `GDP`, `GDP per CAPITA`,`commodity`, `food_supply_stage`, `cause_of_loss`, `treatment`')

    row10_space1, row10_1, row10_space2, row10_2, row10_space3 = st.columns((.1, 1, .1, 1, .1))
    with row10_1:
        st.markdown('### Canada')

        input_year_Canada = [
            st.selectbox('year',
                         [x for x in final_df.query('country == "Canada"').year.unique() if str(x) != 'nan'],
                         key='input_year_Canada')]

        for selection in input_year_Canada:
            selected_input_year_Canada = selection

        input_gdp_Canada = st.text_input('GDP', '',
                                      key='input_gdp_Canada')

        input_gdp_per_capita_Canada = st.text_input('GDP per CAPITA', '',
                                                 key='input_gdp_per_capita_Canada')

        input_commodity_Canada = [
            st.selectbox('commodity',
                         [x for x in final_df.query('country == "Canada"').commodity.unique() if str(x) != 'nan'],
                         key='input_commodity_Canada')]

        for selection in input_commodity_Canada:
            selected_input_commodity_Canada = selection

        input_food_supply_stage_Canada = [
            st.selectbox('food supply stage',
                         [x for x in final_df.query('country == "Canada"').food_supply_stage.unique() if
                          str(x) != 'nan'],
                         key='input_food_supply_stage_Canada')]

        for selection in input_food_supply_stage_Canada:
            selected_input_food_supply_stage_Canada = selection

        input_cause_of_loss_Canada = [
            st.selectbox('cause of loss',
                         [x for x in final_df.query('country == "Canada"').cause_of_loss.unique() if str(x) != 'nan'],
                         key='input_cause_of_loss_Canada')]

        for selection in input_cause_of_loss_Canada:
            selected_input_cause_of_loss_Canada = selection

        input_treatment_Canada = [
            st.selectbox('treatment',
                         [x for x in final_df.query('country == "Canada"').treatment.unique() if str(x) != 'nan'],
                         key='input_treatment_Canada')]

        for selection in input_treatment_Canada:
            selected_input_treatment_Canada = selection

        if st.button('Predict', key='predict_Canada'):
            file_name_Canada = "xgb_Canada.pkl"
            model_Canada = pickle.load(open(file_name_Canada, "rb"))
            data = {'year': [selected_input_year_Canada],
                    'gdp': [input_gdp_Canada],
                    'gdp_per_capita': [input_gdp_per_capita_Canada],
                    'commodity': [selected_input_commodity_Canada],
                    'food_supply_stage': [selected_input_food_supply_stage_Canada],
                    'cause_of_loss': [selected_input_cause_of_loss_Canada],
                    'treatment': [selected_input_treatment_Canada]}
            data = pd.DataFrame(data)
            X = final_df[
                ['year', 'gdp', 'gdp_per_capita', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']]
            X[['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']] = X[
                ['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']].astype('category')
            y = final_df['loss_percentage'].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            encoder = ce.LeaveOneOutEncoder(return_df=True)
            encoder.fit_transform(X_train, y_train)
            row = encoder.transform(data)
            new_data = np.asarray(row.to_numpy())
            yhat = model_Canada.predict(new_data)
            st.write(f'#### Loss percentage predicted:')
            st.write(f'#### {str(yhat).replace("[", "").replace("]", "")}')
        else:
            st.write('#### Please select value to get predicted loss percentage.')

    with row10_2:
        st.markdown('### All countries')
        input_year_all = [
            st.selectbox('year',
                         [x for x in final_df.year.unique() if str(x) != 'nan'],
                         key='input_year_all')]

        for selection in input_year_all:
            selected_input_year_all = selection

        input_gdp_all = st.text_input('GDP', '',
                                      key='input_gdp_all')

        input_gdp_per_capita_all = st.text_input('GDP per CAPITA', '',
                                                 key='input_gdp_per_capita_all')

        input_commodity_all = [
            st.selectbox('commodity',
                         [x for x in final_df.commodity.unique() if str(x) != 'nan'],
                         key='input_commodity_all')]

        for selection in input_commodity_all:
            selected_input_commodity_all = selection

        input_food_supply_stage_all = [
            st.selectbox('food supply stage',
                         [x for x in final_df.food_supply_stage.unique() if
                          str(x) != 'nan'],
                         key='input_food_supply_stage_all')]

        for selection in input_food_supply_stage_all:
            selected_input_food_supply_stage_all = selection

        input_cause_of_loss_all = [
            st.selectbox('cause of loss',
                         [x for x in final_df.cause_of_loss.unique() if str(x) != 'nan'],
                         key='input_cause_of_loss_all')]

        for selection in input_cause_of_loss_all:
            selected_input_cause_of_loss_all = selection

        input_treatment_all = [
            st.selectbox('treatment',
                         [x for x in final_df.treatment.unique() if str(x) != 'nan'],
                         key='input_treatment_all')]

        for selection in input_treatment_all:
            selected_input_treatment_all = selection

        if st.button('Predict', key='predict_all'):
            file_name_all = "xgb_all.pkl"
            model_all = pickle.load(open(file_name_all, "rb"))
            data = {'year': [selected_input_year_all],
                    'gdp': [input_gdp_all],
                    'gdp_per_capita': [input_gdp_per_capita_all],
                    'commodity': [selected_input_commodity_all],
                    'food_supply_stage': [selected_input_food_supply_stage_all],
                    'cause_of_loss': [selected_input_cause_of_loss_all],
                    'treatment': [selected_input_treatment_all]}
            data = pd.DataFrame(data)
            X = final_df[
                ['year', 'gdp', 'gdp_per_capita', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']]
            X[['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']] = X[
                ['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']].astype('category')
            y = final_df['loss_percentage'].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            encoder = ce.LeaveOneOutEncoder(return_df=True)
            encoder.fit_transform(X_train, y_train)
            row = encoder.transform(data)
            new_data = np.asarray(row.to_numpy())
            yhat = model_all.predict(new_data)
            st.write(f'#### Loss percentage predicted:')
            st.write(f'#### {str(yhat).replace("[", "").replace("]", "")}')
        else:
            st.write('#### Please select value to get predicted loss percentage.')

    # st.markdown('### Predict loss percentage by selecting a feature')
    #
    # row10_space1, row10_1, row10_space2, row10_2, row10_space3 = st.columns((.1, 1, .1, 1, .1))
    # dataset = load_csv()
    # X = dataset[['commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']].astype('category')
    # options = ['commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']
    # with row10_1:
    #     st.markdown('### Canada')
    #     file_name_Canada = "xgb_Canada.pkl"
    #     model_Canada = pickle.load(open(file_name_Canada, "rb"))
    #     feature_importance_Canada = model_Canada.feature_importances_
    #     sorted_idx_Canada = np.argsort(feature_importance_Canada)
    #     lookup_Canada = {}
    #     for i in range(0, 4):
    #         lookup_Canada[np.array(X.columns)[sorted_idx_Canada][i]] = feature_importance_Canada[sorted_idx_Canada][i]
    #     option_Canada = st_btn_select(options, index=2, key='feature_importance_Canada')
    #     st.write(f'#### Estimated Loss Percentage:')
    #     st.write(f'#### {lookup_Canada[option_Canada]}')
    #
    # with row10_2:
    #     st.markdown('### All countries')
    #     file_name_all = "xgb_all.pkl"
    #     model_all = pickle.load(open(file_name_all, "rb"))
    #     feature_importance_all = model_all.feature_importances_
    #     sorted_idx_all = np.argsort(feature_importance_all)
    #     lookup_all = {}
    #     for i in range(0, 4):
    #         lookup_all[np.array(X.columns)[sorted_idx_all][i]] = feature_importance_all[sorted_idx_all][i]
    #     option_all = st_btn_select(options, index=2, key='feature_importance_all')
    #     st.write(f'#### Estimated Loss Percentage:')
    #     st.write(f'#### {lookup_all[option_all]}')

    st.markdown('------')
    st.markdown(
        'Appendix A: below section provides the code showing how the figures of feature importance were generated.')
    st.code(code_a, language='python')

    st.markdown(
        'Appendix B: below section provides the code showing how the average loss percentage is predicted.')
    st.code(code_b, language='python')
