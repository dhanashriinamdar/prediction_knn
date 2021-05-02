import numpy as np
import pickle

def valpredict(data):
    loan_amnt = int(data['loan_amnt'])
    int_rate = float(data['int_rate'])
    emp_length = float(data['emp_length'])
    home_ownership = int(data['home_ownership'])
    annual_inc = float(data['annual_inc'])
    dti = float(data['dti'])
    delinq_2yrs = float(data['delinq_2yrs'])
    revol_util = float(data['revol_util'])
    total_acc = float(data['total_acc'])
    longest_credit_length = float(data['longest_credit_length'])
    verified = int(data['verified'])
    months_60 = int(data['months_60'])
    loaded_model = pickle.load(open("model.pkl", "rb"))
    x_predict=[[loan_amnt,int_rate,emp_length,home_ownership,annual_inc,dti,delinq_2yrs,revol_util,total_acc,longest_credit_length,verified,months_60]]
    result = loaded_model.predict(x_predict)
    return result

    


