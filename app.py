from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('fraud_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    txn_id = request.form['transaction_id']
    card = request.form['card_number']
    amount = float(request.form['amount'])

    card_encoded = hash(card) % 10**6
    prediction = model.predict([[card_encoded, amount]])[0]

    if prediction == 1:
        result = f"⚠️ Fraud Detected for Transaction ID: {txn_id}<br>Card: {card}<br>Amount: ₹{amount}"
    else:
        result = f"✅ Transaction ID {txn_id} is Safe."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)