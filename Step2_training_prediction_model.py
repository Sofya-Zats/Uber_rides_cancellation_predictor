# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 18:02:34 2025

@author: SOFYA
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:17:21 2025

@author: SOFYA
"""
#IMPORING PACKAGES


def run():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import GridSearchCV
    

    #1.IMPORING DATA
    data=pd.read_csv("data/bookings_for_cancel_pred_model.csv",index_col=0)
    
    #Cutting out first 20 rows and saving them in separate data frame for later testing
    data_for_testing=data.iloc[:20]
    data_for_testing.to_csv("data/data_for_testing.csv")
    data=data.iloc[20:]
    
    #2.SHUFFLING DATA
    data=shuffle(data,random_state=42)
    
    #3.CHECKING CLASS BALANCE
    #Checking class balance shows that data is not perfectly balanced but not terribly unbalanced either
    class_balance=data["Booking_Status"].value_counts(normalize=True)
    #Result:data is mildly imbalanced: 35%:65%
    
    data.info()
    
    #4.DROPPING COLUMNS THAT HAVE ENCODED COPIES
    data_for_model=data.drop(columns=["Vehicle_Type","Pickup_Location","Drop_Location"])
    
    #3.SPLITTING DATA INTO INPUT VARIABLES AND OUTPUT VARIABLES
    X=data_for_model.drop(columns=["Booking_Status"])
    y=data_for_model["Booking_Status"]
    
    #4.SPLITTING TRAINING AND TEST SETS
    #stratify=y means that y_train and y_test will have same proportion of values "0" and "1"
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                                   random_state=42,stratify=y)
    
    data_for_model.info()
    
    #3.FINDING OPTIMAL PARAMETERS FOR A MODEL
    
    """
    Adding custom class weight to reduce bias toward predicting 1 
    class_weight_ = {
        0: 1,
        1: 0.6   
    }
    Result:model started predicting all outcomes as "1"
    """
    gscv=GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        #estimator=RandomForestClassifier(random_state=42),
        param_grid={  #param_grid=values for n_estimators and for max_depth that we will test to find the nest combination
                      #that produces best accuracy
        "n_estimators": [10, 50, 100, 500],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
    },
        cv=5,
        scoring="r2",#accuracy scores that gridsearch would use
        n_jobs=-1
        )
    
    #Fit to data (note:use gscv.fit on training data)     
    gscv.fit(X,y)
    
    #Get the best cross validation(CV) score (mean)
    gscv.best_score_
    
    #Get optimal parameters that help achive this best CV score
    gscv.best_params_
    
    #Create optimal model object
    clf=gscv.best_estimator_
    
    #5.MODEL TRAINING
    clf.fit(X_train, y_train)
    
    
    
    #6.MODEL ASSESMENT
    #y_pred_class shows prediction for the outcome variable to belong to one of the classes
    #we will only choose a column with probabilities for outcome "1" (customer signing up for a subscription)
    y_pred_class = clf.predict(X_test)
    
    #y_pred_prob shows for each data point probablility of it falling into one of the categories(classes)
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    
    #####Code snippet to adjust threshold
    threshold = 0.5
    y_pred_prob= (y_pred_prob >= threshold).astype(int)
    
    #######
    
    
    #6.1 Plotting confusion matrix
    #conf_matrix shows true/false positives and negatives
    conf_matrix=confusion_matrix(y_test,y_pred_prob)
    #conf_matrix=confusion_matrix(y_test,y_pred_class)
    
    plt.style.use("seaborn-v0_8-poster")
    plt.matshow(conf_matrix, cmap="coolwarm")
    plt.gca().xaxis.tick_bottom()
    plt.style.use("seaborn-v0_8-poster")
    plt.matshow(conf_matrix, cmap = "coolwarm")
    plt.gca().xaxis.tick_bottom()
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class using default threshold=0.5")
    
    for (i, j), con_value in np.ndenumerate(conf_matrix):
        plt.text(j, i, con_value, ha = "center", va = "center", fontsize = 20)
    
    plt.show()
    
   
    pickle.dump(clf,open("model/Cancellations_prediction_model.p", "wb"))
    
    
    
    #6.2 Accuracy score
    #Accuracy score is a proportion of number correct classifications to all the attempted classifications
    #Best for balanced datasets, but can be misleading for imbalanced ones.
    
    accuracy_=accuracy_score(y_test,y_pred_prob)
    #accuracy_=accuracy_score(y_test,y_pred_class)
    
    #6.3 Precision 
    #Precision score is a proportion of all obserations that were predicted as positive to
    # the actual number of positive observations
    #Precision score is animportant metric when the cost of a false positive is high
    
    #precision_=precision_score(y_test,y_pred_class)
    precision_=precision_score(y_test,y_pred_prob)
    
    #6.4 Recall (of all positive obseravtions, how many did we predict as positive)
    #Important when the cost of a false negative is high.
    
    
    #recall_=recall_score(y_test,y_pred_class)
    recall_=recall_score(y_test,y_pred_prob)
    
    #6.5 F-1 (harmonic mean of precision and recall)
    #Provides a balanced measure
    #F-1 is useful when there's an uneven class distribution.
    
    #f1_score_=f1_score(y_test,y_pred_class)
    f1_score_=f1_score(y_test,y_pred_prob)
    
    
    #7.CHECKING FEATURE IMPORTANCES
    
    #7.1 Method feature importance:
    feature_importances=clf.feature_importances_ #Array showing importance of each feature
    feature_importances_df=pd.DataFrame(feature_importances)
    feature_names=pd.DataFrame(X.columns)
    
    feature_importance_summary=pd.concat([feature_names,feature_importances_df],axis=1)
    feature_importance_summary.columns=["feature_name","importance_score"]
    
    
    #Barchart feature-importance
    
    
    plt.barh(feature_importance_summary["feature_name"],feature_importance_summary["importance_score"],color="salmon" )
    plt.title("Feature Importance of Random Forest")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()
    
    
    #7.2 Method permutation importance:
    
    #Repeating permutation(shuffling) 10 times for each feature
    result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)
    
    
    permutation_importances_df=pd.DataFrame(result["importances_mean"])
    feature_names=pd.DataFrame(X.columns)
    permutation_importance_summary=pd.concat([feature_names,permutation_importances_df],axis=1)
    permutation_importance_summary.columns=["feature_name","permutation_importance"]
    permutation_importance_summary=permutation_importance_summary.sort_values(by="permutation_importance")
    
    plt.barh(permutation_importance_summary["feature_name"],permutation_importance_summary["permutation_importance"],color="salmon" )
    plt.title("Permutation Importance of Random Forest")
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    plt.show()
    
    
    #8 CHOOSING OPTIMAL THRESHOLD TO DEAL WITH PRECISION-RECALL TRADE-OFF
    
    #8.1 Plotting precision-recall curve
    from sklearn.metrics import average_precision_score
    
    #y_score = clf.decision_function(X_test)
    average_precision = average_precision_score(y_test,y_pred_class)
    
    #Compute the average precision-recall score
    
    #Average recall looks at how well the model finds positives across different thresholds,
    # giving more importance to thresholds where recall improves the most.
    
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
    
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import PrecisionRecallDisplay
    
    PrecisionRecallDisplay.from_estimator(
        clf,
        X_test,
        y_test
    )
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    data_for_model.info()
    
    #8.2 Using precision-recall curve to find optimal threshold
    pred_score = precision * recall
    knee_idx = np.argmax(pred_score)
    knee_threshold = thresholds[knee_idx]
    
    print("Knee-ish threshold:", knee_threshold)
    print("Precision:", precision[knee_idx])
    print("Recall:", recall[knee_idx])
    
    
    #8.3 Using optimal threshold to refine prediction model
    best_threshold=knee_threshold
    y_pred_prob_refined= (y_pred_prob >= best_threshold).astype(int)
    
    
    #8.4 Plotting confusion matrix for the model with custom threshold
    #conf_matrix shows true/false positives and negatives
    conf_matrix=confusion_matrix(y_test,y_pred_prob_refined)
    #conf_matrix=confusion_matrix(y_test,y_pred_class)
    
    plt.style.use("seaborn-v0_8-poster")
    plt.matshow(conf_matrix, cmap="coolwarm")
    plt.gca().xaxis.tick_bottom()
    plt.style.use("seaborn-v0_8-poster")
    plt.matshow(conf_matrix, cmap = "coolwarm")
    plt.gca().xaxis.tick_bottom()
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel(f"Predicted Class using threshold {best_threshold}")
    
    for (i, j), con_value in np.ndenumerate(conf_matrix):
        plt.text(j, i, con_value, ha = "center", va = "center", fontsize = 20)
    
    plt.show()
    
    
    
    #9 EXPORTING BUNDLE OF OUR TRAINED MODEL PLUS THE BEST THRESHOL AND DATA FOR MODEL.
    import pickle
    
    #Pair: model anmd best threshold
    bundle = {"model": clf, "threshold": float(best_threshold)}
    
    #Exporting pair  it into pickle file
    with open("model/Cancellations_prediction_model_with_threshold.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    #Exporting only a model  without the best threshold
    pickle.dump(clf,open("model/Cancellations_prediction_model.p", "wb"))
    
    #Exporting data for model(data without text columns)
    
    data_for_model.to_csv("data/bookings_for_cancel_pred_model_numeric_col.csv")
    
    #Creating pipeline for our model
    import joblib
    joblib.dump(clf, "Streamlit/model.joblib"  , compress=3)###
    
   
  


 

