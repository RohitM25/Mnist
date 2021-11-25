from flask import Flask,render_template
from flask import request,jsonify
import pickle
import numpy as np

app= Flask(__name__)
model1 = pickle.load(open('salary_model.pkl','rb'))
model2 = pickle.load(open('sales_model.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def predict():
    req = request.json
    mod = req['model_no']
    if mod==1:
        experience = req['experience']
        test_score = req['test_score']
        interview_score = req['interview_score']
        pred = model1.predict([[experience,test_score,interview_score]])
        pred = pred.astype(int).tolist()
        return jsonify({'The predicted salary is':pred[0]} )
    else:
        Store = req['Store']
        DayOfWeek = req['DayOfWeek']
        Customers = req['Customers']
        Open = req['Open']
        Promo = req['Promo']
        StateHoliday = req['StateHoliday']
        SchoolHoliday = req['SchoolHoliday']
        Month = req['Month']
        Year = req['Year']
        Day = req['Day']
        pred = model2.predict([[Store,DayOfWeek,Customers,Open,Promo,StateHoliday,SchoolHoliday,Month,Year,Day]])
        pred = pred.astype(int).tolist()
        return jsonify({'The predicted sales is': pred[0]})





if __name__=='__main__':
    app.run(debug=True,port=9090)
