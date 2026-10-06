from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import pickle
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

app = Flask(__name__)

# Constants
DATA_DIR = "data_set"
DATASET_PATH = os.path.join(DATA_DIR, "flood.csv")
MODELS = {
    "RandomForest": "model_rf.pkl",
    "GradientBoosting": "model_gb.pkl",
    "ANN": "model_ann.pkl"
}
ACTIVE_MODEL_PATH = "flood_model.pkl"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

active_model = None
if os.path.exists(ACTIVE_MODEL_PATH):
    try:
        with open(ACTIVE_MODEL_PATH, "rb") as f:
            active_model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == "admin" and password == "admin":
            return redirect(url_for('home'))
        else:
            return "Invalid credentials.", 401
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/dataset')
def dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        data_html = df.head(100).to_html(classes="styled-table", index=False)
    else:
        data_html = "<p class='error-msg'>Dataset file not found at data_set/flood.csv</p>"
    return render_template('dataset.html', data_table=data_html)

@app.route('/get_random_data')
def get_random_data():
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            random_row = df.drop(columns=['FloodProbability']).sample().to_dict(orient='records')[0]
            return jsonify(random_row)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Dataset not found"}), 404

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    global active_model
    if active_model is None:
        return "No operational model deployed. Please visit the Train panel first.", 500
    
    try:
        feature_names = [
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
            'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
            'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
            'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors'
        ]
        input_data = [float(request.form[feature]) for feature in feature_names]
        prediction_val = active_model.predict([input_data])[0]
        return render_template('prediction_result.html', prediction=round(float(prediction_val), 3))
    except Exception as e:
        return f"Prediction Error: {str(e)}", 400

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    global active_model
    try:
        if not os.path.exists(DATASET_PATH):
            return "Dataset missing!", 404
            
        df = pd.read_csv(DATASET_PATH)
        
        # SPEED OPTIMIZATION: Train on a random subset of 5,000 rows
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
            
        X = df.drop(columns=['FloodProbability']) 
        y = df['FloodProbability']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        algos = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            ),
            "ANN": make_pipeline(
                StandardScaler(), 
                MLPRegressor(
                    hidden_layer_sizes=(50, 25), alpha=0.5, early_stopping=True, 
                    max_iter=200, random_state=42
                )
            )
        }
        
        results = {}
        best_r2 = -float('inf')
        best_algo_model = None
        
        for name, algo in algos.items():
            algo.fit(X_train, y_train)
            r2 = r2_score(y_test, algo.predict(X_test))
            accuracy_pct = round(max(0.0, r2 * 100), 2)
            results[name] = accuracy_pct
            
            with open(MODELS[name], "wb") as f:
                pickle.dump(algo, f)
                
            if r2 > best_r2:
                best_r2 = r2
                best_algo_model = algo

        if best_algo_model:
            with open(ACTIVE_MODEL_PATH, "wb") as f:
                pickle.dump(best_algo_model, f)
            active_model = best_algo_model
            
        return render_template('train_result.html', results=results, best_score=round(best_r2*100, 2))
    except Exception as e:
        return f"Training Failure: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)