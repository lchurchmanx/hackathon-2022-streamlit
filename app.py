import os

from app_content import *

# Set page title and favicon.
st.set_page_config(
    page_title="Loan Suggestion",
    layout="wide"
)

menu = st.sidebar.selectbox(
    "Select from the dropdown menu to explore",
    ["Intro",
     "Data Preview",
     "Loan Score",
     "Data Quality",
     # "Data Quality",
     "Feature Importance",
     "Data Exploration",
     # "Data Map Plot",
     # "Data Relationship",
     # "Feature Importance and Prediction"
     ],
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

if menu == 'Intro':
    set_home()
elif menu == 'Data Preview':
    preview_data()
elif menu == 'Data Quality':
    set_data_quality()
# elif menu == 'Data Overview':
#     set_data_overview()
elif menu == 'Data Exploration':
    set_data_exploration()
elif menu == 'Feature Importance':
    set_feature_importance()
# elif menu == 'Data Relationship':
#     set_relationship()
# elif menu == 'Feature Importance and Prediction':
#     set_feature_importance()
elif menu == 'Loan Score':
    loan_scoring()
else:
    in_progress()
