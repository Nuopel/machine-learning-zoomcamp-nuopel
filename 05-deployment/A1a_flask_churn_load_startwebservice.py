import pickle
from flask import Flask, request, jsonify


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('churn_model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    print('run in a terminal :\n python3 A1a_flask_churn_load_startwebservice.py for dev mode  \n or gunicorn --bind 0.0.0.0:9696 A1a_flask_churn_load_startwebservice:app for full mode')
    # in dev mode in a terminal at /path where is A1a_flask_churn_load_startwebservice.py
    # python3 A1a_flask_churn_load_startwebservice.py

    # run in deploy mode ( interminal but with the correct venv associated ):
    # gunicorn --bind 0.0.0.0:9696 A1a_flask_churn_load_startwebservice:app