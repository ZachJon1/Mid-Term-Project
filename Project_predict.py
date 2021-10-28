import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_file.bin'

with open(model_file, 'rb') as model_in:
    dv, model = pickle.load(model_in)

app = Flask('rating')

@app.route('/predict', methods=['POST'] )

def predict():
    info = request.get_json()

    X =dv.transform([info])
    y_pred=model.predict(X)
   

    result = {
        'Book rating': float(y_pred)
    }

    return jsonify(result)
  



if __name__ == "__main__":
    app.run(debug=True, host= '0.0.0.0', port=9696)