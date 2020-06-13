""" Importing other scripts"""
from file_operations import file_methods
from preprocessing import preprocessing
from Application_Logger import logger
from FeatureEngineering import Feature_Selection
""" Loading libraries for training the model """

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.simplefilter("ignore")


class train:

    def __init__(self):
        self.file_object = open("Training_Logs/ModelTrainingLogs.txt",'a+')
        self.file_ops_file_object = open("Training_Logs/File_Operations_Logs.txt",'a+')
        self.feature_file_object = open("Training_Logs/FeatureSelectionLogs.txt",'+a')
        self.log_writer = logger.App_Logger()

    def modeling(self, df):
        try:
            self.log_writer.log(self.file_object, "------------------------------------------------------------------")
            self.log_writer.log(self.file_object,"start of the training")
            self.log_writer.log(self.file_object,"Loading data")
            self.df = df

            """ Performing data preprocessing """
            self.log_writer.log(self.file_object,"Preprocessing training data")
            preprocessor = preprocessing.preprocessor(self.file_object, self.log_writer)
            self.df = preprocessor.preprocess_pipeline(self.df)

            """Train Test Split"""
            self.log_writer.log(self.file_object,"Train Test Split")
            X = self.df.drop(columns=['class'])
            y = self.df['class']


            # best_cols = ['Attr6', 'Attr29', 'Attr27', 'Attr25', 'Attr55', 'Attr5', 'Attr39', 'Attr3',
            #              'Attr24', 'Attr9', 'Attr57', 'Attr51', 'Attr41', 'Attr38', 'Attr35', 'Attr32',
            #              'Attr2', 'Attr16', 'Attr10', 'Attr7', 'Attr58', 'Attr47', 'Attr46', 'Attr28',
            #              'Attr26', 'Attr22', 'Attr18', 'Attr14', 'Attr12', 'Attr1']
            best_cols = ['Attr6', 'Attr29', 'Attr27', 'Attr25', 'Attr55', 'Attr5', 'Attr39', 'Attr38',
                         'Attr35', 'Attr24', 'Attr51', 'Attr32', 'Attr3', 'Attr2', 'Attr16', 'Attr12',
                         'Attr10', 'Attr9', 'Attr7', 'Attr58', 'Attr57', 'Attr47', 'Attr46', 'Attr28',
                         'Attr26', 'Attr22', 'Attr18', 'Attr15', 'Attr14', 'Attr11']
            X = X[best_cols]

            X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, random_state=0, test_size=0.3)

            """Selecting best 30 features for training"""
            fs = Feature_Selection.Features(self.feature_file_object, X_train, y_train, 30)
            best_features = fs.ensemble_select()
            print(best_features)

            # """ Model fitting """
            self.log_writer.log(self.file_object,"Defining classifier")
            xgb = XGBClassifier(max_depth=9, min_child_weight=1, gamma=0.4, subsample=0.9, colsample_bytree=0.7,
                                reg_alpha=0.1)


            self.log_writer.log(self.file_object,"Fitting the classifier onto Training data")

            xgb = xgb.fit(X_train, y_train)


            """ Hyperparameter Tuning"""
            # self.log_writer.log(self.file_object, "Model Hyperparameter tuning of XGBOOST ")
            # params = {
            # 'max_depth': range(3, 10, 2),
            # 'min_child_weight': range(1, 6, 2),
            # 'subsample': [i / 10.0 for i in range(6, 10)],
            # 'colsample_bytree': [i / 10.0 for i in range(6, 10)]
            # }
            # # xgb = XGBClassifier(max_depth=9,min_child_weight=1, gamma=0.4, subsample=0.9, colsample_bytree=0.7, reg_alpha=0.1)
            # grid = GridSearchCV(XGBClassifier(gamma=0.4,reg_alpha=0.1), params, scoring='f1', verbose=2, n_jobs=-1)
            # grid.fit(X_train, y_train)
            # print(grid.best_estimator_)
            # self.log_writer.log(self.file_object, "Model Hyperparameter tuning of XGBOOST Successful ")


            """ Saving model """
            self.log_writer.log(self.file_object,"Saving the trained model")
            file = file_methods.File_Operations(self.file_ops_file_object)
            file.save_model(model=xgb, model_name='xgb')
            # file.save_model(model=grid, model_name="xgb_tuning")

            """ Loading saved model """
            self.log_writer.log(self.file_object,"Loading the saved model")
            model = file.load_model('xgb')

            self.log_writer.log(self.file_object,"Predicting the test data")
            y_pred = model.predict(X_test)

            print("Accuracy Score : {}".format(accuracy_score(y_test, y_pred)))
            print("F1 Score : {}".format(f1_score(y_test, y_pred)))
            print("AUC SCore : {}".format(roc_auc_score(y_test, y_pred)))
            self.log_writer.log(self.file_object,"Training Successful !")
            self.file_object.close()
        except Exception as e:
            self.file_object.close()
            raise Exception()




#
# if __name__=='__main__':
#     from Training import train
#     df = pd.read_csv("../data.csv")
#     tr = train.train()
#     tr.modeling(df)
