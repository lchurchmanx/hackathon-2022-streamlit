path_bank_dataset = './data/ETHOS_INGESTION_AUDIT_BANK_TOT_TRANS.csv'
path_merchant_dataset = './data/ETHOS_INGESTION_AUDIT_MERCH_TOT_TRANS.csv'
path_ranking_dataset = './data/ranking_data.parquet'
path_modeling_merchant_dataset = './data/data_modeling_merchant.parquet'
path_merge_data_dataset = './data/merge_data.parquet'
path_scoring_dataset = './data/scoring_report_output.csv'
path_agg = './data/agg_data.csv'
path_merchant = './data/merchant_data.csv'
path_banking = './data/banking_data.csv'
path_feature_importance = './data/feature_importance_output.csv'

LOAN_SMALL = 'Less than 20K'
LOAN_MEDIUM = '20- 50K'
LOAN_LARGE = '50K-120K'

MERCHANT_SUBSET = ["529000243688", "4445024841130", "4445026810325"]
# MERCHANT_SUBSET = ["4445000986976", "4445024609222", "529000243688"] # OLD ONES

intro = '''
## Bringing the FIS Merchant (Vantiv/WorldPay ) and FIS Issuing together with advanced data analytics
'''

intro_context = '''

The main goals:
* Simplify the customer experience
* Ability for Banks and FI's to provide customized and personalized loan offers to merchants.
* Easy integration with current FIS offerings.
* Targeted advertising potential for Financial institutions.
* Use machine learning algorithms to score merchants without additional documentation.

### Where should I go from here?
Investigate data behind score calculation with Data Preview, Data Quality, and Data Exploration. 
View Feature Importance tab to learn about our method. 
Use the Loan Score to determine a given merchant's score.
'''