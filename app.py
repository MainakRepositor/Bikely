from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

st.title("Bike Rental Rate Prediction")

data = pd.read_csv("bikes.csv")
data1 = data.copy()
label_encoder = preprocessing.LabelEncoder()
lr = LinearRegression()
Z = data[['base_price','kms_driven','age','power','stroke','milage','length','weight','acceleration','brand']]
lr.fit(Z, data['rent'])



nav = st.sidebar.radio("Sections",["Home","Prediction"])

if nav == "Home":
    st.image("bike.jpg",width= 700)
    if st.checkbox("Show Dataset"):
        st.dataframe(data1)
    
    st.header("Want to know how the feature effects the Price ?")
    feature = st.selectbox("Select a feature", ["power","age","kms_driven","stroke"])
    
    if feature == "power":
        st.subheader("Horsepower vs price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["power"],data["rent"])
        plt.ylim(0)
        plt.xlabel("Horsepower")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "age":
        st.subheader("Age vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["age"],data["rent"])
        plt.ylim(0)
        plt.xlabel("Age")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "kms_driven":
        st.subheader("Kilometers Driven vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["kms_driven"],data["rent"])
        plt.ylim(0)
        plt.xlabel("Kilometers Driven")
        plt.ylabel("Price")
        st.pyplot(ax)

    elif feature == "stroke":
        st.subheader("Stroke vs Price")
        ax = plt.figure(figsize= (10,5))
        plt.scatter(data["stroke"],data["rent"])
        plt.ylim(0)
        plt.xlabel("Stroke")
        plt.ylabel("Price")
        st.pyplot(ax)            




elif nav == "Prediction":
    st.image("money.jpg",width= 800)
    st.header("Estimate your Bike Rent")
    
    base_price = st.slider("Base Price (in Rs.)", float(data["base_price"].min()), float(data["base_price"].max()))
    kms_driven = st.slider("kms driven", int(data["kms_driven"].min()), int(data["kms_driven"].max()))
    age = st.slider("Age", int(data["age"].min()), int(data["age"].max()))
    power = st.slider("Horsepower", int(data["power"].min()), int(data["power"].max()))
    stroke = st.slider("Number of strokes", int(data["stroke"].min()), int(data["stroke"].max()))
    milage = st.slider("Milage", int(data["milage"].min()), int(data["milage"].max()))
    length = st.slider("Length", int(data["length"].min()), int(data["length"].max()))
    weight = st.slider("Weight", int(data["weight"].min()), int(data["weight"].max()))
    acceleration = st.slider("Acceleration", int(data["acceleration"].min()), int(data["acceleration"].max()))
    brand = st.slider("Bike Brand", int(data["brand"].min()), int(data["brand"].max()))

    val = np.array([base_price,kms_driven,age,power,stroke,milage,length,weight,acceleration,brand]).reshape(1,-1)
    
    data_brands = pd.read_csv('bike_brands.csv')
    st.sidebar.info("Bike Brands Available")
    st.sidebar.table(data_brands['brand'].unique())


    pred = lr.predict(val)[0]
    pred = round(pred)
    if st.button("Predict"):
        st.success(f"Your predicted bike share price for this month is Rs. {pred}")


