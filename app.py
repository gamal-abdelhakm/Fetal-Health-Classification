import joblib
from flask import Flask, render_template, request 

app = Flask(__name__)
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')
@app.route('/', methods=['GET'])

def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET'])
def get_predict():
    inp_data = [
        request.args.get("accelerations") ,
        request.args.get("uterine_contractions") ,
        request.args.get("prolongued_decelerations") ,
        request.args.get("abnormal_short_term_variability") ,
        request.args.get("percentage_of_time_with_abnormal_long_term_variability") ,
        request.args.get("mean_value_of_long_term_variability") ,
        request.args.get("histogram_mode") ,
        request.args.get("histogram_mean") ,
        request.args.get("histogram_median") ,
        request.args.get("histogram_variance") 
        ]
    fetal_health_type = model.predict(scaler.transform([inp_data]))[0]

    if fetal_health_type == 1:
        fetal_health_type = 'Normal'
    elif fetal_health_type == 2:
        fetal_health_type = 'Suspect'
    elif fetal_health_type == 3:
        fetal_health_type = 'Pathological'

    return render_template("index.html" , fetal_health_type = fetal_health_type)

if __name__ == '__main__' :
    app.run(debug = True,host="127.0.0.15")