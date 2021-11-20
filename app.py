from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

st.title("Used Motorbike Price Prediction")

data = pd.read_csv("bikes.csv")
data1 = data.copy()
label_encoder = preprocessing.LabelEncoder()
data['brand']= label_encoder.fit_transform(data['brand'])
data['brand'].unique()
lr = LinearRegression()
Z = data[['power', 'age', 'kms_driven','stroke','brand']]
lr.fit(Z, data['price'])



nav = st.sidebar.radio("Sections",["Home","Prediction"])
st.sidebar.markdown("Made by [Mainak](https://www.linkedin.com/in/mainak-chaudhuri-127898176/)")  
if nav == "Home":
    st.image("bike.jpg",width= 700)
    if st.checkbox("Show Dataset"):
        st.dataframe(data1)
    
    st.header("Want to know how the feature effects the Price ?")
    feature = st.selectbox("Select a feature", ["power","age","kms_driven","stroke"])
    
    if feature == "power":
        st.subheader("Horsepower vs price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["power"],data["price"])
        plt.ylim(0)
        plt.xlabel("Horsepower")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "age":
        st.subheader("Age vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["age"],data["price"])
        plt.ylim(0)
        plt.xlabel("Age")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "kms_driven":
        st.subheader("Kilometers Driven vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["kms_driven"],data["price"])
        plt.ylim(0)
        plt.xlabel("Kilometers Driven")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "stroke":
        st.subheader("Stroke vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["stroke"],data["price"])
        plt.ylim(0)
        plt.xlabel("Stroke")
        plt.ylabel("Price")
        st.pyplot(ax)            




elif nav == "Prediction":
    st.image("money.jpg",width= 500)
    st.header("Know your Bike Price")
    val1 = st.number_input("Enter your Bike's Horsepower",100,2000,step = 3)
    val2 = st.number_input("Enter your Bike's Age (years)",1,40,step = 5)
    val3 = st.number_input("Enter your Bike's Kilometers Travelled (in 1000 km)",1,50,step = 2)
    val4 = st.number_input("Enter your Bike's Stroke",1,5,step = 1)
    val5 = st.number_input("Enter your Bike's Brand",0,22,step = 1)

    val = np.array([val1,val2,val3,val4,val5]).reshape(1,-1)
    
    datax = [data1["brand"], data["brand"]]
    headers = ["data1", "data"]
    df3 = pd.concat(datax, axis=1, keys=headers)
    dfx = df3.sort_values('data')
    dfx = dfx.drop_duplicates()
    dfx.reset_index(drop=True, inplace=True)
    st.markdown("### Choose a brand id from the brand list given below, by pressing on the + button")
    dfx.rename(columns={'data1': 'Bike Brand','data':'Brand ID'}, inplace=True)
    st.dataframe(dfx)


    pred = lr.predict(val)[0]
    pred = round(pred)
    if st.button("Predict"):
        st.success(f"Your predicted motorbike price is Rs. {pred}")


