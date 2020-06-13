""" Importing other scripts"""
from file_operations import file_methods
from preprocessing import preprocessing
from Application_Logger import logger

""" Loading libraries required for prediction """
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np


class predict:

    def __init__(self):
        self.file_object = open("Prediction_Logs/PredictionLogs.txt",'a+')
        self.file_ops_file_object = open("Prediction_Logs/File_Operations_Logs.txt", 'a+')
        self.feature_file_object = open("Prediction_Logs/FeatureSelectionLogs.txt", '+a')
        self.log_writer = logger.App_Logger()

    def predictions(self, df):
        self.log_writer.log(self.file_object, "------------------------------------------------------------------")
        self.log_writer.log(self.file_object,"start of the Prediction")
        self.log_writer.log(self.file_object,"Loading test data")
        self.df = df

        print(self.df['class'].value_counts())

        """ Doing data preprocessing steps on test data"""
        self.log_writer.log(self.file_object,"Preprocessing test data")
        preprocessor = preprocessing.preprocessor(self.file_object, self.log_writer)

        self.df = preprocessor.preprocess_pipeline_pred(self.df)

        best_cols = ['Attr6', 'Attr29', 'Attr27', 'Attr25', 'Attr55', 'Attr5', 'Attr39', 'Attr38',
                     'Attr35', 'Attr24', 'Attr51', 'Attr32', 'Attr3', 'Attr2', 'Attr16', 'Attr12',
                     'Attr10', 'Attr9', 'Attr7', 'Attr58', 'Attr57', 'Attr47', 'Attr46', 'Attr28',
                     'Attr26', 'Attr22', 'Attr18', 'Attr15', 'Attr14', 'Attr11']

        """input and label Split"""
        self.log_writer.log(self.file_object,"Train Test Split")
        X_test = self.df.drop(columns=['class'])
        y_test = self.df['class']


        file = file_methods.File_Operations(self.file_ops_file_object)
        self.log_writer.log(self.file_object,"Loading the saved model")
        model = file.load_model('xgb')


        self.log_writer.log(self.file_object,"Predicting the test data")
        y_pred = model.predict(X_test[best_cols])

        print("Predicted Value Counts")
        unique, counts = np.unique(y_pred, return_counts=True)
        print(np.asarray((unique, counts)).T)

        print("Original value_counts")
        unique, counts = np.unique(y_test, return_counts=True)
        print(np.asarray((unique, counts)).T)

        print("Accuracy Score : {}".format(accuracy_score(y_test, y_pred)))
        print("F1 Score : {}".format(f1_score(y_test, y_pred)))
        print("AUC SCore : {}".format(roc_auc_score(y_test, y_pred)))
        self.log_writer.log(self.file_object,"Prediction Successful !")
        return pd.DataFrame(data=y_pred, columns=['Prediction'], index=X_test.index)

# if __name__=='__main__':
#     from Prediction import predict
#     # df = pd.read_csv("../Dataset/testing.csv")
#     df = pd.read_csv("Dataset/test.csv")
#     tr = predict.predict()
#     tr.predictions(df)