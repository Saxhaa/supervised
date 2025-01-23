import numpy as np
import matplotlib.pyplot as plt
import scipy 
import sklearn 
import pandas as pd
import csv
import joblib
import sys

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


if __name__=="__main__":

    tab_model = ["RandomForest_BestModel_08196.joblib", "AdaBoost_BestModel_08150.joblib", "XGB_BestModel_08248.joblib"]

    split_number = sys.argv[1]
    model = tab_model[int(sys.argv[2])]

    df_attributes = pd.read_csv(f'db_oral/features_split_{split_number}.csv')
    df_labels = pd.read_csv(f'db_oral/labels_split_{split_number}.csv')



    scaler = joblib.load('scaler.joblib')
    columns_to_standardize = ["AGEP", "WKHP"]

    df_attributes_scaled=df_attributes.copy()
  
    df_attributes_scaled[columns_to_standardize] = scaler.transform(df_attributes[columns_to_standardize])


                         
    def dataset_analysis(df):
        plt.hist(df["AGEP"], bins=20, density=True)
        # plt.hist(df["RAC1P"], density=True)
        # count = df_attributes.SEX.value_counts()
        # labels = ['Male', 'Female']
        # plt.pie(count, labels=labels)
        # plt.hist(df_attributes["COW"], density=True)
        # plt.hist(df_attributes["SCHL"], density=True)
        # plt.hist(df_attributes["MAR"], density=True)
        # plt.hist(df_attributes["OCCP"], density=True)
        # plt.hist(df_attributes["POBP"], density=True)
        # plt.hist(df_attributes["RELP"], density=True)
        # plt.hist(df_attributes["WKHP"], density=True)
        plt.show()



    rf_best = joblib.load(model)
    labels_pred = rf_best.predict(df_attributes_scaled)

    accuracy = accuracy_score(df_labels, labels_pred)
    f1 = f1_score(df_labels, labels_pred)

    print("Accuracy: ", accuracy, "\nF1 Score: ", f1)

    #disp = ConfusionMatrixDisplay.from_predictions(df_labels, labels_pred, normalize="pred")
    #disp.plot()
    #plt.show()


    # debug
    # print("coucou")