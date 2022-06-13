path_dataset = './data/Data_World.csv'
path_gdp = './data/gdp.csv'
path_gdp_per_capita = './data/gdppc.csv'

intro = '''
# Background of the Food Loss and Waste Database
##### Sustainable Development Goal Indicator 12.3 states “By 2030, to halve per capita global food waste at the retail and consumer levels and reduce food losses along production and supply chains, including post-harvest losses.”
##### To help monitor the state of food loss, FAO conducted an extensive review of literature in the public domain which gathered data and information from almost 500 publications, reports, and studies from various sources (including from organizations like the World Bank, GIZ, FAO, IFPRI, and more).
##### The data from this review is held within the interactive Food Loss and Waste Database which allows for micro and macro analysis of different sets of data.
'''


intro_context = '''
## Food Loss and Waste Database
#### Take an in-depth look at what food is being lost and wasted, and where
##### The Food Loss and Waste database is the largest online collection of data on both food loss and food waste and causes reported throughout the literature. The database contains data and information from openly accessible reports and studies measuring food loss and waste across food products, stages of the value chain, and geographical areas. In October 2019, more than 480 publications and reports from various sources (e.g., subnational reports, academic studies, and reports from national and international organizations such as the World Bank, GIZ, FAO, IFPRI, and other sources), which have produced more than 20 thousand data points, were included. Data can be queried, downloaded, and plotted in an interactive and structured way. The database can be used by anyone who wishes to know more about food losses and waste.

###### cited from [fao.org](https://www.fao.org/platform-food-loss-waste/flw-data/en/)
'''

column_table = '''
| Column Name | Description |
| ----------- | ----------- |
| m49_code | Desc. |
| country | Desc. |
| region | Desc. |
| cpc_code | Desc. |
| commodity | Desc. |
| year | Desc. |
| loss_percentage | Desc. |
| loss_percentage_original | Desc. |
| loss_quantity | Desc. |
| activity | Desc. |
| food_supply_stage | Desc. |
| treatment | Desc. |
| cause_of_loss | Desc. |
| sample_size | Desc. |
| method_data_collection | Desc. |
| reference | Desc. |
| url | Desc. |
| notes | Desc. |
'''

code_a = '''
food_waste_df = pd.read_csv('Data_World.csv')
gdp_df = pd.read_csv('gdp.csv')
gdppc_df = pd.read_csv('gdppc.csv')

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

X = final_df[['year', 'gdp', 'gdp_per_capita', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']]
X[['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']] = X[['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']].astype('category')
y = final_df['loss_percentage'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
encoder = ce.LeaveOneOutEncoder(return_df=True)
X_train_New = encoder.fit_transform(X_train, y_train)
X_test_New = encoder.transform(X_test)
model = XGBRegressor(n_estimators=500, max_depth=5, eta=0.05)
model.fit(X_train_New, y_train)
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)'''

code_b = '''
X = dataset[['year', 'gdp', 'gdp_per_capita', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']]
X[['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']] = X[['year', 'commodity', 'food_supply_stage', 'cause_of_loss', 'treatment']].astype('category')
y = dataset['loss_percentage'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
encoder = ce.LeaveOneOutEncoder(return_df=True)
encoder.fit_transform(X_train, y_train)
row = encoder.transform(data)
new_data = asarray(row.to_numpy())
yhat = model_Canada.predict(new_data)'''