# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 15:47:49 2025

@author: SOFYA
"""

import pandas as pd

def run():
    print("Loading data...")
    bookings=pd.read_csv("data/Bookings.csv")
    
    bookings.info()
    
    
    #STEP 0:EXPLORATORY ANALYSIS OF DATA SET
    bookings["Booking_ID"].nunique() 
    
    bookings["Booking_Status"].value_counts() 
    
    bookings["Vehicle_Type"].value_counts() 
    
    
    bookings["Pickup_Location"].value_counts() 
    bookings["Drop_Location"].value_counts() 
    
    
    
    
    bookings["Canceled_Rides_by_Customer"].value_counts() 
    
    bookings["Canceled_Rides_by_Driver"].value_counts() 
    
    bookings["Incomplete_Rides_Reason"].value_counts() 
    
    bookings["Customer_Rating"].value_counts() 
    
    
    #Checking percentage of unknown values in Driver_Ratings and Customer_Rating columns
    Driver_Rating_empty_perc=bookings["Driver_Ratings"].isna().sum()/bookings["Driver_Ratings"].size*100
    print(f"Unknown values for driver rating occupy {Driver_Rating_empty_perc:.2f} % of all values")
    
    Customer_Rating_empty_perc=bookings["Customer_Rating"].isna().sum()/bookings["Customer_Rating"].size*100
    print(f"Unknown values for customer rating occupy {Customer_Rating_empty_perc:.2f} % of all values")
    
    
    #Checking what % of  data frame consists of rows where
    # both driver rating and customer rating are not empty
    rating_known=bookings.dropna(subset=["Customer_Rating","Driver_Ratings" ])
    rating_known_perc=rating_known.size/bookings.size*100
    print(f"Rows with knownvalues for customer rating and driver  occupy {rating_known_perc:.2f} % of all values")
    
    #STEP 1:CLEANING DATA
    
    #1.1 Convert time variables into numerical variables
    bookings["Day"]=pd.to_datetime(bookings["Date"]).dt.day
    bookings["Month"]=pd.to_datetime(bookings["Date"]).dt.month
    bookings["Day_of_week"]=pd.to_datetime(bookings["Date"]).dt.dayofweek
    
    bookings["Hour"]=pd.to_datetime(bookings["Time"]).dt.hour
    
    
    #1.2 Convert text variables into numerical variables
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    bookings["Vehicle_Type_encoded"] = le.fit_transform(bookings["Vehicle_Type"] )
    bookings["Pickup_Location_encoded"] = le.fit_transform(bookings["Pickup_Location"] )
    bookings["Drop_Location_encoded"] = le.fit_transform(bookings["Drop_Location"] )
    
    bookings_with_text_columns=bookings.copy(deep=True)
    
    
    # "Booking_Status" is output column for our cancellation prediction model.
    #     Filling it with "0" for succesfully requested rides and "1" for cancelled rides
    
    bookings_for_cancel_pred_model=bookings.loc[~(bookings["Booking_Status"]=="Driver Not Found")]
    bookings_for_cancel_pred_model["Booking_Status"]=bookings_for_cancel_pred_model["Booking_Status"].replace("Success",0)
    bookings_for_cancel_pred_model["Booking_Status"]=bookings_for_cancel_pred_model["Booking_Status"].replace("Canceled by Customer",1)
    bookings_for_cancel_pred_model["Booking_Status"]=bookings_for_cancel_pred_model["Booking_Status"].replace("Canceled by Driver",1)
    
    
    
    #1.3 For rides that were booked sucesfully but were cancelled later: setting "Booking_status" to value 0 
    booked_but_incomplete=bookings_for_cancel_pred_model.loc[ (bookings_for_cancel_pred_model["Booking_Status"]==0) & (bookings_for_cancel_pred_model["Incomplete_Rides"]=="Yes")   ]
    
    bookings_for_cancel_pred_model.loc[
        booked_but_incomplete.index, "Booking_Status"
    ] = 1
    
    
    
    #1.4 Removing columns that do not affect probability of cancellation
    bookings_for_cancel_pred_model=bookings_for_cancel_pred_model.drop(columns=[
                                    "Date","Time","Vehicle Images","Unnamed: 20",
                                    "Canceled_Rides_by_Customer",
                                    "Canceled_Rides_by_Driver","Customer_ID",
                                    "Incomplete_Rides_Reason",
                                    "Payment_Method","Incomplete_Rides",
                                    "Booking_ID",
                                    "V_TAT","C_TAT",
                                    "Driver_Ratings",
                                    "Customer_Rating"
                                    ])
    
    bookings_with_text_columns.to_csv("data/bookings_with_text_columns.csv")
    bookings_for_cancel_pred_model.to_csv("data/bookings_for_cancel_pred_model.csv")