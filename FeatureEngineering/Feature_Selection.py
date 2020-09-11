"""Import from other scripts"""
from Application_Logger import logger
from preprocessing import preprocessing

"""Import from libraries"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

class Features:
    """
        This class shall  be used to for selecting best performing features.

        Written By: Tushar Chaudhari
        Version: 1.0
        Revisions: None
    """
    def __init__(self,file_object, X,y, n):
        self.file_object = file_object
        self.log_writer = logger.App_Logger()
        self.features = list(X.columns)
        self.num_features = n
        self.X = X
        self.y = y
        self.n = n

    """Pearson Corellation"""
    def Pearson_Corr(self):
        """
                Method Name: Pearson_Corr
                Description: This is a filter-based method. We check the absolute value of the Pearson's correlation
                            between the target and numerical features in our dataset. We keep the top n features based
                             on this criterion.
                Output: Boolean list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered Pearson_Corr method of the Features class in Feature_selection")
        self.corr_list = []
        feature_name = self.features
        try:
            for i in feature_name:
                cor = np.corrcoef(self.X[i], self.y)[0,1]
                self.corr_list.append(cor)
            self.corr_list = [0 if np.isnan(i) else i for i in self.corr_list]
            self.corr_feature = self.X.iloc[:, np.argsort(np.abs(self.corr_list))[-self.num_features:]].columns.tolist()
            self.corr_support = [True if i in self.corr_feature else False for i in feature_name]
            self.log_writer.log(self.file_object,"Pearson_Corr method successfull !")
            return  self.corr_support
        except Exception as e:
            self.log_writer.log(self.file_object,"Exception occured in Pearson_Corr method pf Preprocessor class. Exception message: " + str(e))
            self.log_writer.log(self.file_object, "Pearson_Corr method for feature selection unsuccessful")
            raise Exception()



    """Chi-Square test"""
    def chi_square(self):
        """
                Method Name: chi_square
                Description: This is filter-based feature selection method.  In this method, we calculate the
                                chi-square metric between the target and the numerical variable and
                                only select the variable with the maximum chi-squared values.
                Output: Boolean list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered chi_square method of the Features class in Feature_selection")
        try:
            X_norm =MinMaxScaler().fit_transform(self.X)
            self.chi_selector = SelectKBest(chi2, k = self.n)
            self.chi_selector.fit(X_norm, self.y)
            self.chi_support = self.chi_selector.get_support()
            self.chi_feature = self.X.loc[:, self.chi_support].columns.tolist()
            self.log_writer.log(self.file_object,"chi_square method for feature selection successfull !")
            return self.chi_support
        except Exception as e:
            self.log_writer.log(self.file_object,
                                "Exception occured in chi_square method pf Preprocessor class. Exception message: " + str(
                                    e))
            self.log_writer.log(self.file_object, "chi_square method for feature selection unsuccessful")
            raise Exception()


    """Recursive Feature Elimination method"""
    def recursive_feature_elimination(self):
        """
                Method Name: recursive_feature_elimination
                Description: This is a wrapper based feature selection method. Recursive feature elimination (RFE)
                            select features by recursively considering smaller and smaller sets of features. First,
                            the estimator is trained on the initial set of features and the importance of each feature
                            is obtained either through a coef_ attribute or through a feature importances attribute. Then,
                            the least important features are pruned from current set of features. That procedure is
                            recursively repeated on the pruned set until the desired number of features to select is
                            eventually reached.
                Output: Boolean list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered recursive_feature_elimination method of the Features class in Feature_selection")
        try:
            X_norm = MinMaxScaler().fit_transform(self.X)
            self.rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=self.n, step=10, verbose=5)
            self.rfe_selector.fit(X_norm, self.y)
            self.rfe_support = self.rfe_selector.get_support()
            self.rfe_feature = self.X.loc[:, self.rfe_support].columns.tolist()
            self.log_writer.log(self.file_object, "Recursive Feature Elimination method for feature selection successfull !")
            return  self.rfe_support
        except Exception as e:
            self.log_writer.log(self.file_object,
                                "Exception occured in recursive_feature_elimination method pf Preprocessor class. Exception message: " + str(
                                    e))
            self.log_writer.log(self.file_object, "recursive_feature_elimination method for feature selection unsuccessful")
            raise Exception()


    """ Linear Feature Selection method """
    def lasso_selection(self):
        """
                Method Name: lasso_selection
                Description: This is an Embedded method for feature selection. Embedded methods use algorithms that
                            have built-in feature selection methods.  For example, Lasso, and RF have their own
                            feature selection methods. Lasso Regularizer forces a lot of feature weights to be zero.
                            Here we use Lasso to select variables.
                Output: Boolean list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered  lasso_selection method of the Features class in Feature_selection")
        try:
            X_norm = MinMaxScaler().fit_transform(self.X)
            self.embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=self.n)
            self.embeded_lr_selector.fit(X_norm, self.y)
            self.embeded_lr_support = self.embeded_lr_selector.get_support()
            self.embeded_lr_feature = self.X.loc[:, self.embeded_lr_support].columns.tolist()
            self.log_writer.log(self.file_object, "Lasso selection method for feature selection successfull !")
            return self.embeded_lr_support
        except Exception as e:
            self.log_writer.log(self.file_object,
                                "Exception occured in lasso_selection method pf Preprocessor class. Exception message: " + str(
                                    e))
            self.log_writer.log(self.file_object, "lasso_selection method for feature selection unsuccessful")
            raise Exception()

    """"Tree based feature Selection method(Random Forest)"""
    def rf_based_selection(self):
        """
                Method Name: rf_based_selection
                Description: This is an Embedded method. Here RandomForest is used to select features based on feature
                            importance. We calculate feature importance using node impurities in each decision tree.
                            In Random forest, the final feature importance is the average of all decision tree
                            feature importance.
                Output: Boolean list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered rf_based_selection method of the Features class in Feature_selection")
        try:
            self.embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=self.n)
            self.embeded_rf_selector.fit(self.X, self.y)
            self.embeded_rf_support = self.embeded_rf_selector.get_support()
            self.log_writer.log(self.file_object, "Random Forest based selection method for feature selection successfull !")
            return self.embeded_rf_support
        except Exception as e:
            self.log_writer.log(self.file_object,"Exception occured in rf_based_selection method pf Preprocessor class. Exception message: " + str(e))
            self.log_writer.log(self.file_object, "rf_based_selection method for feature selection unsuccessful")
            raise Exception()


    """"Tree based feature Selection method(LightGBM)"""
    def lgb_based_selection(self):
        """
                Method Name: lgb_based_selection
                Description: This is an Embedded method. Here LightGBM is used to select features based on feature
                            importance. We calculate feature importance using node impurities in each decision tree.
                            In Random forest, the final feature importance is the average of all decision tree
                            feature importance.
                Output: Boolean list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered lgb_based_selection method of the Features class in Feature_selection")
        try:
            lgbc = LGBMClassifier()
            self.embeded_lgb_selector = SelectFromModel(lgbc, max_features=self.n)
            self.embeded_lgb_selector.fit(self.X, self.y)
            self.embeded_lgb_support = self.embeded_lgb_selector.get_support()
            self.log_writer.log(self.file_object,"LightGBM based selection method for feature selection successfull !")
            return self.embeded_lgb_support
        except Exception as e:
            self.log_writer.log(self.file_object,"Exception occured in lgb_based_selection method pf Preprocessor class. Exception message: " + str(e))
            self.log_writer.log(self.file_object, "lgb_based_selection method for feature selection unsuccessful")
            raise Exception()



    """ combbination of all feature selection method """
    def ensemble_select(self):
        """
                Method Name: ensemble_select
                Description: This method aggregrates the output from above all feature selection methods, arranges the
                            features commonly occuring in all feature selection methods.
                Output: list of Top n best features
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "--" * 20 + "Start" + "--" * 20)
        self.log_writer.log(self.file_object, "Entered ensemble_select method of the Features class in Feature_selection")
        try:
            pd.set_option('display.max_rows', None)
            corr_support = self.Pearson_Corr()
            chi_support = self.chi_square()
            rfe_support = self.recursive_feature_elimination()
            embeded_lr_support = self.lasso_selection()
            embeded_rf_support = self.rf_based_selection()
            embeded_lgb_support = self.lgb_based_selection()

            # put all selection together
            feature_selection_df = pd.DataFrame(
                {'Feature': self.features, 'Pearson': corr_support, 'Chi-2': chi_support, 'RFE': rfe_support,
                 'Logistics': embeded_lr_support,
                 'Random Forest': embeded_rf_support, 'LightGBM': embeded_lgb_support}
            )
            # count the selected times for each feature
            feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
            feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
            feature_selection_df.index = range(1, len(feature_selection_df) + 1)
            print(feature_selection_df.Feature[:30].values)
            self.log_writer.log(self.file_object, "Feature selection successfull !")
            self.log_writer.log(self.file_object,"--" * 20 + "End" + "--" * 20 )
            return feature_selection_df.Feature[:30]

        except Exception as e:
            self.log_writer.log(self.file_object, "Feature selection unsuccessful")
            raise Exception()


# if __name__ == "__main__":
#
#     df = pd.read_csv('G:/iNeuron/Pycharm/1year.csv')
#     file = open("../temp/temp.txt", 'a+')
#     log_writer = logger.App_Logger()
#     preprocessor = preprocessing.preprocessor(file, log_writer)
#     # Removing  '?' from data
#     df = preprocessor.replace_special_character(df)
#     # Removing unwanted spaces from the data
#     df = preprocessor.remove_unwanted_spaces(df)
#     # Converting numerical data from object datatype to integer datatype
#     df = preprocessor.obj_to_num(df)
#     # Scaling numerical variables using standard scaler
#     df = preprocessor.scaling_numerical_columns(df)
#     # Downsampling
#     df = preprocessor.downsampling(df)
#
#     df = df.fillna(df.mean())
#     X = df.drop(columns=['class'])
#     y = df['class']
#     fs = Features(X,y, 30)
#     print(X.isna().sum())
#     fs.ensemble_select()
