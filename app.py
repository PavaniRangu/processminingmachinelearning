from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Focal loss function
def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce)
    focal_loss = alpha * (1 - bce_exp) ** gamma * bce
    return focal_loss

# Load model and scaler
model = tf.keras.models.load_model('heart_disease_prediction_model.h5', custom_objects={"focal_loss_fixed": focal_loss_fixed})
scaler = pickle.load(open('scaler.pkl', 'rb'))

features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age', 'Education', 'Income'
]

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[feature]) for feature in features]
        final_features = np.array([input_features])
        final_features = scaler.transform(final_features)
        prediction = model.predict(final_features)
        output = (prediction > 0.5).astype(int)
        result = 'High Risk of Heart Disease 😟' if output[0][0] == 1 else 'Low Risk (Healthy) 😊'
        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error occurred: {str(e)}"

@app.route('/get_hospitals', methods=['POST'])
def get_hospitals():
    city = request.form['city']
    hospitals_data = {
        "Andhra Pradesh": [{"name": "Apollo Health City, Hyderabad", "phone": "040-23607777"}],
        "Arunachal Pradesh": [{"name": "TRIHMS, Naharlagun", "phone": "0360-2351471"}],
        "Assam": [{"name": "Guwahati Medical College", "phone": "0361-2130020"}],
        "Bihar": [{"name": "Indira Gandhi Institute of Medical Sciences, Patna", "phone": "0612-2297099"}],
        "Chhattisgarh": [{"name": "AIIMS Raipur", "phone": "0771-2573777"}],
        "Goa": [{"name": "Goa Medical College", "phone": "0832-2495000"}],
        "Gujarat": [{"name": "Civil Hospital Ahmedabad", "phone": "079-22681060"}],
        "Haryana": [{"name": "PGIMS Rohtak", "phone": "01262-211300"}],
        "Himachal Pradesh": [{"name": "IGMC Shimla", "phone": "0177-2658831"}],
        "Jharkhand": [{"name": "RIMS Ranchi", "phone": "0651-2541533"}],
        "Karnataka": [{"name": "Narayana Health, Bengaluru", "phone": "1860-208-0208"}],
        "Kerala": [{"name": "Amrita Hospital, Kochi", "phone": "0484-2851234"}],
        "Madhya Pradesh": [{"name": "AIIMS Bhopal", "phone": "0755-2672333"}],
        "Maharashtra": [{"name": "KEM Hospital, Mumbai", "phone": "022-24136051"}],
        "Manipur": [{"name": "JNIMS Imphal", "phone": "0385-2414629"}],
        "Meghalaya": [{"name": "NEIGRIHMS Shillong", "phone": "0364-2538013"}],
        "Mizoram": [{"name": "Civil Hospital Aizawl", "phone": "0389-2305393"}],
        "Nagaland": [{"name": "Naga Hospital Kohima", "phone": "0370-2222957"}],
        "Odisha": [{"name": "SCB Medical College, Cuttack", "phone": "0671-2414304"}],
        "Punjab": [{"name": "Dayanand Medical College, Ludhiana", "phone": "0161-4686600"}],
        "Rajasthan": [{"name": "SMS Medical College, Jaipur", "phone": "0141-2518291"}],
        "Sikkim": [{"name": "STNM Hospital, Gangtok", "phone": "03592-202515"}],
        "Tamil Nadu": [{"name": "Apollo Hospitals, Chennai", "phone": "044-28293333"}],
        "Telangana": [{"name": "Yashoda Hospitals, Hyderabad", "phone": "040-45674567"}],
        "Tripura": [{"name": "Agartala Government Medical College", "phone": "0381-2353000"}],
        "Uttar Pradesh": [{"name": "SGPGIMS Lucknow", "phone": "0522-2668700"}],
        "Uttarakhand": [{"name": "AIIMS Rishikesh", "phone": "0135-2462925"}],
        "West Bengal": [{"name": "NRS Medical College, Kolkata", "phone": "033-22653500"}],
        "Andaman and Nicobar Islands": [{"name": "GB Pant Hospital, Port Blair", "phone": "03192-232102"}],
        "Chandigarh": [{"name": "PGIMER Chandigarh", "phone": "0172-2747585"}],
        "Dadra and Nagar Haveli and Daman and Diu": [{"name": "Government Hospital, Silvassa", "phone": "0260-2642940"}],
        "Delhi": [{"name": "AIIMS Delhi", "phone": "011-26588500"}],
        "Jammu and Kashmir": [{"name": "SKIMS Srinagar", "phone": "0194-2401013"}],
        "Ladakh": [{"name": "SNM Hospital, Leh", "phone": "01982-252118"}],
        "Lakshadweep": [{"name": "Government Hospital, Kavaratti", "phone": "04896-262264"}],
        "Puducherry": [{"name": "JIPMER Puducherry", "phone": "0413-2298288"}]
    }

    hospitals = hospitals_data.get(city, [])
    return render_template('hospitals.html', city=city, hospitals=hospitals)

if __name__ == "__main__":
    app.run(debug=True)
