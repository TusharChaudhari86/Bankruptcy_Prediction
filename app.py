from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import pandas as pd
from werkzeug import secure_filename
from Prediction.predict import predict
from Training.train import train
import warnings
warnings.simplefilter("ignore")

app = Flask(__name__)
CORS(app)

@app.route("/", methods = ['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict_custom", methods=['POST'])
@cross_origin()
def custom_predict():
    try:
        path = 'Uploads/test.csv'
        df = pd.read_csv(path)
        pred = predict()
        pred = pred.predictions(df)
        pred.to_csv("Downloads/Predictions.csv")
        os.remove("Uploads/test.csv")
        return Response("Prediction File created and ready to download, "
                        +"Predicted file path "+str(path))
    except ValueError:
        return Response("Value Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("KeyError Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Exception Error Occurred! %s" %e)


@app.route("/predict_default", methods=['POST'])
@cross_origin()
def default_predict():
    try:
        path = 'Dataset/test.csv'
        df = pd.read_csv(path)
        pred = predict()
        pred = pred.predictions(df)
        pred.to_csv("Downloads/Predictions.csv")
        return Response("Prediction File created and ready to download, "
                        +"Predicted file path ")
    except ValueError:
        return Response("Value Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("KeyError Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Exception Error Occurred! %s" %e)



@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    try:
       if request.method == 'POST':
          file = request.files['file']
          if file :
              filename = secure_filename(file.filename)
              file.save(os.path.join('Uploads', filename))
              os.rename(os.path.join('Uploads', filename),os.path.join('Uploads', "test.csv"))
          return Response("Upload successfull!!")
    except Exception as e:
        print(e)
        raise Exception()

@app.route("/train", methods=['GET'])
@cross_origin()
def trainRouteClient():
    try:
        df = pd.read_csv("data.csv")
        training = train()
        training.modeling(df)

    except ValueError:
        return Response("Value Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("KeyError Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Exception Error Occurred! %s" %e)

    return Response("Training successfull!!")


port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    app.run(port=port,debug=True)
