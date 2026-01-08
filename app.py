

"""
Created on Mon Dec 22 21:15:38 2025

@author: SOFYA
"""

import streamlit as st
import pandas as pd
import joblib
import datetime
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



d = st.date_input("When was the ride requested?", datetime.date(2019, 7, 6))
Day=d.day
Month=d.month
Day_of_week=d.weekday()


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
   

    proba = round(model.predict_proba(input_df)[0][1],2)
    probability=round(model.predict_proba(input_df)[0][0],2)
    pred = int(proba >= 0.5)  # threshold can be changed
    prediction=""
    if pred==0:
            prediction="Most likely this ride won't be cancelled"
    elif pred==1:
           prediction="Most likely this ride will be cancelled" 
    st.subheader("Result")
    st.write(f"Prediction: **{prediction}**")

    st.write(f"Probability of ride being cancelled: **{proba:.3f}**")
    

