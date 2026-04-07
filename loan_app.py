import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from imblearn 




# page configuration
st.set_page_config('Loan Model',layout ='wide')


# loading pre-trained model
with open ('best_model.pkl','rb')as f:
    loan_model =joblib.load(f)

# loading the scaler
with open('standard_scaler.pkl','rb')as f:
    scaler =joblib.load(f)

scaler =StandardScaler()

# loading feature importance from an excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# loading ther feature importance dataframe
final_feature_imp =load_feature_importance('feature_importance.xlsx')

# setting up streamlit sidebar
image_sidebar =Image.open('loan2.jpeg')
st.sidebar.image(image_sidebar, width="stretch")
st.sidebar.header('LOAN FEATURES')


# split layout into two columns
# left_col, right_col =st.sidebar.columns(2)
# gathering the input 
# feature selection on sidebar
def get_user_input():
    left_col, right_col =st.sidebar.columns(2)
    with left_col:
        Age	 =st.sidebar.number_input('Age(No.)',min_value=18.0,max_value=90.0,step=1.0,value=43.0)
        Income =st.sidebar.number_input('Income(No.)',min_value=15000.0, max_value=149999.0, step=0.5, value=82466.0)
        LoanAmount =st.sidebar.number_input('Loan_Amount(No.)',min_value=5000.0,max_value=249999.0,step=0.5,value=127556.0)
        CreditScore	 =st.sidebar.number_input('Credit_Score(No.)',min_value=300.0,max_value=849.0,step=0.5,value=574.0)
        MonthsEmployed =st.sidebar.number_input('Months_Employed(No.)',min_value=0.0, max_value=119.0,step=0.5,value=60.0)
        NumCreditLines =st.sidebar.number_input('Num_Credit_Lines(No.)',min_value=1.0,max_value=4.0,step=0.5,value=2.0)
        InterestRate =st.sidebar.number_input('Interest_Rate(No.)',min_value=2.0,max_value=25.0,step=0.5,value=13.32)
        LoanTerm =st.sidebar.number_input('Loan_Term (No.)',min_value=12.0,max_value=60.0,step=0.5,value=36.3)
        DTIRatio =st.sidebar.number_input('DTI_Ratio(No.)',min_value=0.1,max_value=0.9,step=0.1,value=0.5)
    
    with right_col:
        Education =st.sidebar.selectbox('Education ',["Bachelor's", 'High School', "Master's", 'PhD'])
        EmploymentType =st.sidebar.selectbox('Employment_Type',['Full-time', 'Part-time' ,'Self-employed', 'Unemployed'])
        MaritalStatus =st.sidebar.selectbox('Marital_Status',['Divorced' ,'Married' ,'Single'])
        HasMortgage =st.sidebar.selectbox('Has_Mortgage',['Yes','No'])
        HasDependents =st.sidebar.selectbox('Has_Dependents',['No','Yes']) 
        LoanPurpose =st.sidebar.selectbox('LoanPurpose',['Auto', 'Business' ,'Education', 'Home', 'Other'])
        HasCoSigner =st.sidebar.selectbox('HasCoSigner',['No','Yes'])
        LTI_Rate = LoanAmount/Income
        FinRiskScore_FRS = CreditScore - LTI_Rate + (MonthsEmployed / 12)

    user_data={
        Age	:Age,
        Income :Income,
        LoanAmount :LoanAmount,
        CreditScore :CreditScore,
        MonthsEmployed :MonthsEmployed,
        NumCreditLines :NumCreditLines,
        InterestRate :InterestRate,
        LoanTerm :LoanTerm,
        DTIRatio :DTIRatio,
        f'Education{Education}':1,
        f'EmploymentType{EmploymentType}': 1,
        f'MaritalStatus{MaritalStatus}' : 1,
        f'HasMortgage{HasMortgage}' : 1,
        f'HasDependents{HasDependents}': 1,
        f'LoanPurpose{LoanPurpose}':1,
        f'HasCoSigner{HasCoSigner}' :1,
        LTI_Rate :LTI_Rate, #this compute the load to income rate of the 
        FinRiskScore_FRS:FinRiskScore_FRS #this compute financial riest
    }
    return user_data


# user_data =get_user_input()


# adding main header image
impage_banner =Image.open('loan1.jpeg')
st.image(impage_banner, width="stretch")

# centerizing title
st.markdown("<h1 style = 'text-align:center;'>Loan prediction app</>", unsafe_allow_html=True)


# split layout into two columns
left_col, right_col =st.columns(2)


# "left columns: contain feature importance interactive bar chart"
with left_col:
    st.header("Feature Importance")


    # # sorting freature importance dataframe by 'feature importance score 
    final_fi_sorted =final_feature_imp.sort_values(by='feature importance score:', ascending=True)
    
    # variable	feature importance score:
    # create intaractive bar chart with plotly
    fig =px.bar(
    final_fi_sorted,
    x='feature importance score:',
    y='variable',
    orientation ='h',
    title='Feature Importance',
    labels={'feature importance score:':'importance','variable':'feature'},
    text ='feature importance score:',
    color_discrete_sequence=['#48a3b4']
    )
    
    fig.update_layout(
    xaxis_title ='Feature importance score:',
    yaxis_title ='variable',
    template ='plotly_white',
    height =500
    )
    
    # setting the streamlit graph function
    st.plotly_chart(fig,use_column_width=True)


# right column: prediction interface
with right_col:
    st.header("Predict Loan Defaulters")


    # getting user inputs from sidebar
    user_data =get_user_input()


    # tranform the input into the required format
    def prepare_input(data, feature_list):
        input_data ={feature: data.get(feature,0) for feature in feature_list}
        return np.array([list(input_data.values())])
    
    # feature list (same order as use during model train)
    features =[
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm','DTIRatio', 'HasMortgage', 'HasDependents', 'HasCoSigner', 'Education_Bachelors', 'Education_High School','Education_Masters', 'Education_PhD','EmploymentType_Full time', 'EmploymentType_Part-time','EmploymentType_Self employed', 'EmploymentType_Unemployed','MaritalStatus_Divorced', 'MaritalStatus_Married','MaritalStatus_Single', 'LoanPurpose_Auto', 'LoanPurpose_Business','LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other','LTI_Rate', 'FinRiskScore_FRS'
    ]
    
    
    # predict button
    if st.button('Predict'):
         input_array =prepare_input(user_data,features)
         scale_input =scaler.fit_transform(input_array)
         prediction =loan_model.predict(scale_input)
         st.subheader('Predicted loan defaulter:')
         if prediction==1:
            st.write(f'will Default')
         else:
            st.write(f"will not Default")



