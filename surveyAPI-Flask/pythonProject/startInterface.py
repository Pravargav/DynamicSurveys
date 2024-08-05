import torch
import sys
sys.path
sys.path.append(r'C:\Users\dell\surveyApi\pythonProject')
from flask_cors import CORS
from main2 import model_gru,make_pred



symp2 = "I've been itching a lot, and it's been accompanied with a rash that looks to be getting worse over time. \
There are also some patches of skin that are different colours from the rest of the skin,\
as well as some lumps that resemble little nodes."
model_gru.load_state_dict(torch.load(r'C:\Users\dell\PycharmProjectfx\pythonProject\modelx\modex.pt'))
model_gru.eval()
val=make_pred(model_gru, symp2)

import joblib
modelMe2 = joblib.load( r'C:\Users\dell\surveyApi\pythonProject\modelx\model2x.pkl')
import numpy as np
single_row = np.array([[1 ,0 , 1  ,0  ,23  ,0  ,2  ,2,0]])
from main import modelMe
print(modelMe.predict(single_row))

from flask import Flask,request,jsonify
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>hello world!</p>"+str(val)

@app.route("/mine",methods=['GET'])
def hello_world2():
    data=request.args.get('text')
    val3 = make_pred(model_gru, data)
    resp={'valx':val3}
    return jsonify(resp)

@app.route("/mine2",methods=['GET'])
def hello_world3():
    checkbox1 = request.args.get('checkbox1')
    checkbox2 = request.args.get('checkbox2')
    checkbox3 = request.args.get('checkbox3')
    checkbox4 = request.args.get('checkbox4')

    select1 = request.args.get('select1')
    sel1=int(select1)
    select2 = request.args.get('select2')
    sel2=int(select2)
    select3 = request.args.get('select3')
    sel3=int(select3)
    select4 = request.args.get('select4')
    sel4=int(select4)

    integral=request.args.get('integral')

    row=np.array([[checkbox1,checkbox2,checkbox3,checkbox4,integral,sel1,sel2,sel3,sel4]])
    val4= modelMe.predict(row)
    resp={'valx':val4[0]}
    return jsonify(resp)

if __name__ == "__main__":
    app.run()