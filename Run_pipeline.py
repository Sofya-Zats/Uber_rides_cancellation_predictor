# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 18:28:49 2025

@author: SOFYA
"""

from Step1_cleaning_data import run as load_data
from Step2_training_prediction_model import run as train_prediction_model
from Step3_deploying_model_to_streamlit import run as deploy_model_to_streamlit

def main():
    load_data()
    train_prediction_model()
    deploy_model_to_streamlit()

if __name__ == "__main__":
    main()