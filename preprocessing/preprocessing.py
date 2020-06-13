import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler

class preprocessor:
    """
        This class shall  be used to clean and transform the data before training or predicting.

        Written By: Tushar Chaudhari
        Version: 1.0
        Revisions: None
    """
    def __init__(self, file_object, log_writer):
        self.file_object = file_object
        self.log_writer = log_writer


    def obj_to_num(self,df):
        """
                Method Name: obj_to_num
                Description: This method corrects the datatype of object to float of a pandas dataframe.
                Output: A pandas DataFrame after converting the datatype.
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.df = df
        self.log_writer.log(self.file_object,"Entered object to num method of the Preprocessor class")
        try:
            self.log_writer.log(self.file_object, "Conversion of the dtype of variables of dataset successful")
            for col in self.df.columns:
                if self.df[col].dtype == 'O':
                    self.df[col] = pd.to_numeric(self.df[col])
            self.log_writer.log(self.file_object, "Datatype conversion Successful")
            return self.df
        except Exception as e:
            self.log_writer.log(self.file_object, "Exception occured in obj_to_num method pf Preprocessor class. Exception message: "+str(e))
            self.log_writer.log(self.file_object,"Conversion of datatype unsuccessful")
            raise Exception()


    def replace_special_character(self,df):
        """
                Method Name: replace_special_character
                Description: This method replace the special character "?" with "NaN" in pandas dataframe.
                Output: A pandas DataFrame with clean data.
                On Failure: Raise Exception

                Written By: Tushar Chaudhari
                Version: 1.0
                Revisions: None
        """
        self.log_writer.log(self.file_object, "Entered replace_special_character method of the Preprocessor class ")
        self.df = df
        try:
            self.log_writer.log(self.file_object, "Removing ? from the dataframe columns")
            return self.df.replace('?', np.NaN)
            self.log_writer.log(self.file_object, "Removing ? from the dataframe columns Successful")
        except Exception:
            self.logger_object.log(self.file_object,'Exception occured in replace_special_character method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'removal of special characters Unsuccessful. Exited the replace_special_character method of the Preprocessor class')
            raise Exception()


    def remove_unwanted_spaces(self,df):
        """
                        Method Name: remove_unwanted_spaces
                        Description: This method removes the unwanted spaces pandas dataframe.
                        Output: A pandas DataFrame with clean data.
                        On Failure: Raise Exception

                        Written By: Tushar Chaudhari
                        Version: 1.0
                        Revisions: None
        """
        self.log_writer.log(self.file_object,"Entered remove_unwanted_spaces method of the Preprocessor class ")
        self.df = df
        try:
            self.df_without_spaces=self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            self.log_writer.log(self.file_object, "unwanted space removal successful. Excited the remove_unwanted_spaces method of the Preprocessor class.")
            return self.df_without_spaces

        except Exception as e:
            self.log_writer.log(self.file_object,'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object,'unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()


    def scaling_numerical_columns(self, df):
        """
                        Method Name: scaling_numerical_columns
                        Description: This method apply robust scaling on the numerical varibles of the pandas dataframe.
                        Output: A pandas DataFrame with scaled data.
                        On Failure: Raise Exception

                        Written By: Tushar Chaudhari
                        Version: 1.0
                        Revisions: None
        """
        self.log_writer.log(self.file_object,'Entered the scaling_numerical_columns method of the Preprocessor class')
        self.df = df
        try:
            self.log_writer.log(self.file_object, "Applying Robust scaling on the Numerical variables of the dataframe")
            sc = RobustScaler()
            X = self.df.drop(columns=['class'])
            y = self.df['class']
            X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
            self.log_writer.log(self.file_object, 'Scaling for numerical values successful. Exited the scaling_numerical_columns method of the Preprocessor class')
            return pd.concat([X,y], axis=1)
        except Exception as e:
            self.log_writer.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object, 'Scaling for numerical variables Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()


    def upsampling(self, df):
        """
                        Method Name: upsampling
                        Description: This method upsamples the minority class data (creates duplicates of observations) in the pandas dataframe.
                        Output: A pandas DataFrame with upsampled data.
                        On Failure: Raise Exception

                        Written By: Tushar Chaudhari
                        Version: 1.0
                        Revisions: None
        """
        self.log_writer.log(self.file_object, 'Entered the upsampling method of the Preprocessor class')
        self.df = df

        try:
            data = self.df
            data = data.sample(frac=1)
            df_major = data.loc[data['class'] == 0]
            df_minor = data.loc[data['class'] == 1]
            df_minority_upsampled = resample(df_minor,
                                             replace=True,  # sample with replacement
                                             n_samples=len(df_major),  # to match majority class
                                             random_state=123)  # reproducible results

            data =  pd.concat([df_major, df_minority_upsampled])
            self.log_writer.log(self.file_object,'Upsampling if the monority class data successful. Exited the upsampling method of the Preprocessor class')
            return data
        except Exception as e:
            self.log_writer.log(self.file_object,'Exception occured in upsampling method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object,'Upsampling of the minority data Failed. Exited the upsampling method of the Preprocessor class')
            raise Exception()


    def downsampling(self, df):
        """
                        Method Name: downsampling
                        Description: This method downsamples the majority class data (creates duplicates of observations) in the pandas dataframe.
                        Output: A pandas DataFrame with scaled data.
                        On Failure: Raise Exception

                        Written By: Tushar Chaudhari
                        Version: 1.0
                        Revisions: None
        """
        self.log_writer.log(self.file_object, 'Entered the downsampling method of the Preprocessor class')
        self.df = df
        try:
            df = self.df.sample(frac=1)
            minor = df.loc[df['class'] == 1]
            major = df.loc[df['class'] == 0][:df['class'].value_counts()[1]]
            normal_distributed_df = pd.concat([minor, major])
            self.log_writer.log(self.file_object, 'Downsampling if the monority data successful. Exited the upsampling method of the Preprocessor class')
            return normal_distributed_df

        except Exception as e:
            self.log_writer.log(self.file_object,'Exception occured in Downsampling method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object,'Downsampling of the minority data Failed. Exited the Downsampling method of the Preprocessor class')
            raise Exception()


    def preprocess_pipeline(self, df):
        """
                        Method Name: preprocess_pipeline
                        Description: This method calls other preprocessing methods sequentially such that output of one method is input to other method.
                        Output: A pandas DataFrame with preprocessed data.
                        On Failure: Raise Exception

                        Written By: Tushar Chaudhari
                        Version: 1.0
                        Revisions: None
                        Note : This method should not be applied on test data.
        """
        try:
            self.df = df
            self.df = self.replace_special_character(self.df)
            self.df = self.obj_to_num(self.df)
            self.df = self.scaling_numerical_columns(self.df)
            self.df = self.df.fillna(0)
            self.df = self.upsampling(self.df)
            return self.df
        except Exception as e:
            self.log_writer.log(self.file_object,'Exception occured in preprocess_pipeline method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object,'Preprocessing on the train data Failed. Exited the preprocess_pipeline method of the Preprocessor class')
            raise Exception()

    def preprocess_pipeline_pred(self, df):
        """
                        Method Name: preprocess_pipeline
                        Description: This method is supposed to be used on test data, which calls relevent preprocessing methods sequentially such that output of one method is input to other method.
                        Output: A pandas DataFrame with preprocessed data.
                        On Failure: Raise Exception

                        Written By: Tushar Chaudhari
                        Version: 1.0
                        Revisions: None
                        Note : This method is supposed to be used only on test data.
        """
        try:
            self.df = df
            self.df = self.replace_special_character(self.df)
            self.df = self.obj_to_num(self.df)
            self.df = self.scaling_numerical_columns(self.df)
            self.df = self.df.fillna(0)
            return self.df
        except Exception as e:
            self.log_writer.log(self.file_object,'Exception occured in preprocess_pipeline method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object,'Preprocessing on the test data Failed. Exited the preprocess_pipeline_pred method of the Preprocessor class')
            raise Exception()





if __name__ == "__main__":
    from Application_Logger import logger
    df = pd.read_csv('G:/iNeuron/Pycharm/1year.csv')
    file_object = open("../Training_Logs/ModelTrainingLogs.txt", 'a+')
    log_writer = logger.App_Logger()
    p = preprocessor(file_object,log_writer)
    df = p.remove_unwanted_spaces(df)
    df = p.replace_special_character(df)
    df = p.obj_to_num(df)
    df = p.scaling_numerical_columns(df)
    df = p.upsampling(df)

    print(df['class'].value_counts())