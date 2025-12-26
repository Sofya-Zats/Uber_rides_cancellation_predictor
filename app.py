

"""
Created on Mon Dec 22 21:15:38 2025

@author: SOFYA
"""

import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Prediction App", layout="centered")

#UPLOADING OUR PRETRAINED MODEL
@st.cache_resource
def load_model():
    return joblib.load("Streamlit/model.joblib") 

model = load_model()

#ADDING INTERFACE ELEMENTS TO THE STREAMLIT PAGE

#UPLOADING FILE WITH ORIGINAL TEXT COLUMNS (BEFORE WE ENCODED THEM INTO NUMERICAL FORMAT)
bookings_with_text_columns=pd.read_csv("data/bookings_with_text_columns.csv",index_col=0)
bookings_with_text_columns.info()



st.title("Ride cancellation predictor")
st.write("Enter inputs below:")

Booking_Value = st.number_input("Booking_Value (numeric)", value=549)
Ride_Distance = st.number_input("Ride_Distance (numeric)", value=15.7)


 
i=1
days=[]
while i<32:
    days.append(i) 
    i=i+1
Day= st.selectbox("Day:", (days))


Month_str= st.selectbox("Month:", ("January","February","March","April",
                               "May","June","July","August","September",
                               "October","November","December")  )

if Month_str=="January": Month=1
elif Month_str=="February" :Month=2
elif Month_str=="March" :Month=3
elif Month_str=="April" :Month=4
elif Month_str=="May" :Month=5
elif Month_str=="June" :Month=6
elif Month_str=="July" :Month=7
elif Month_str=="August" :Month=8
elif Month_str=="September" :Month=9
elif Month_str=="October" :Month=10
elif Month_str=="November" :Month=11
elif Month_str=="December" :Month=12           
     
    

Day_of_week_str= st.selectbox("Day_of_week", ("Monday","Tuesday","Wednesday",
                                          "Thursday","Friday",
                                          "Saturday","Sunday"))
if Day_of_week_str=="Monday": Day_of_week=0
elif Day_of_week_str=="Tuesday" :Day_of_week=1
elif Day_of_week_str=="Wednesday" :Day_of_week=2
elif Day_of_week_str=="Thursday" :Day_of_week=3
elif Day_of_week_str=="Friday" :Day_of_week=4
elif Day_of_week_str=="Saturday" :Day_of_week=5
elif Day_of_week_str=="Sunday" :Day_of_week=6


h=0
hours=[]
while h<24:
    hours.append(h) 
    h=h+1
Hour= st.selectbox("Hour", hours)  
   

Vehicle_Type_str= st.selectbox("Vehicle_Type",("Prime Sedan","eBike","Auto","Prime Plus","Bike","Prime SUV","Mini" )    )
if Vehicle_Type_str=="Auto":Vehicle_Type=0
elif Vehicle_Type_str=="Bike" :Vehicle_Type=1
elif Vehicle_Type_str=="Mini" :Vehicle_Type=2
elif Vehicle_Type_str=="Prime Plus" :Vehicle_Type=3
elif Vehicle_Type_str=="Prime SUV" :Vehicle_Type=4
elif Vehicle_Type_str=="Prime Sedan" :Vehicle_Type=5
elif Vehicle_Type_str=="eBike" :Vehicle_Type=6


#Extracting pairs:pick up location name-corresponding numerical encoding
Pick_up_locations=bookings_with_text_columns[["Pickup_Location","Pickup_Location_encoded"]].drop_duplicates(["Pickup_Location","Pickup_Location_encoded"])

Pick_up_locations_list=Pick_up_locations.values.tolist()
Pickup_Location_str= st.selectbox("Pickup_Location",[l[0] for l in Pick_up_locations_list])
for l in Pick_up_locations_list:
    if Pickup_Location_str==l[0]:
          Pickup_Location=l[1] 

#Extracting pairs:drop location name-corresponding numerical encoding
Drop_locations=bookings_with_text_columns[["Drop_Location","Drop_Location_encoded"]].drop_duplicates(["Drop_Location","Drop_Location_encoded"])

Drop_locations_list=Drop_locations.values.tolist()
Drop_Location_str= st.selectbox("Drop_Location",[l[0] for l in Drop_locations_list])
for l in Drop_locations_list:
    if Drop_Location_str==l[0]:
          Drop_Location=l[1] 


   
# Building one-row DataFrame with axact column names used in training
input_df = pd.DataFrame([{
    "Booking_Value": Booking_Value,
    "Ride_Distance": Ride_Distance,
    "Day": Day,
    "Month":Month,
    "Day_of_week":Day_of_week,
    "Hour":Hour,
    "Vehicle_Type_encoded":Vehicle_Type,
    "Pickup_Location_encoded":Pickup_Location,
    "Drop_Location_encoded":Drop_Location
      
}])

#Making prediction based on the input 
if st.button("Predict"):
   
    proba = model.predict_proba(input_df)[0][1]
    pred = int(proba >= 0.5)  # threshold can be changed
    prediction=""
    if pred==0:
        prediction="most likely this ride will be cancelled"
    elif pred==1:
       prediction="most likely this ride won't be cancelled" 
    st.subheader("Result")
    st.write(f"Prediction: **{prediction}**")

