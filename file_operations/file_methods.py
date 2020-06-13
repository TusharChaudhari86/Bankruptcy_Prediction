""" Loading files for file operations """
import pickle

"""Import from other scripts"""
from Application_Logger import logger



class File_Operations:

    def __init__(self, file_object):
        self.file_object = file_object
        self.log_writer = logger.App_Logger()

    def save_model(self,  model, model_name):
        self.log_writer.log(self.file_object, "Entered save_model method of the File Operations class")
        try:
            path = "models/"
            pickle.dump(model, open(path + model_name +".sav", 'wb'))
            print('Model Saved Successfully')
            self.log_writer.log(self.file_object,"Model Saved successfully in models directory")
        except Exception as e:
            self.log_writer.log(self.file_object,"Exception occured in save_model method of File Operations class. Exception message: " + str(e))
            self.log_writer.log(self.file_object, "save_model method for File Operations unsuccessful")
            raise Exception()


    def load_model(self, model_name):
        self.log_writer.log(self.file_object, "Entered load_model method of the File Operations class")
        try:
            path = "models/" + model_name + ".sav"
            model = pickle.load(open(path, 'rb'))
            self.log_writer.log(self.file_object,"Model loaded successfully from models directory")
            return model
        except Exception as e:
            self.log_writer.log(self.file_object,"Exception occured in load_model method of File Operations class. Exception message: " + str(e))
            self.log_writer.log(self.file_object, "load_model method for File Operations unsuccessful")
            raise Exception()


