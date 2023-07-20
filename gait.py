from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the sensor data from the CSV file
sensor_data = pd.read_csv('sensor_data.csv', index_col=0)

# Perform K-means clustering
kmeans = KMeans(n_clusters=8)  # Assuming 8 different gait characteristics
clusters = kmeans.fit_predict(sensor_data)
sensor_data['Gait Characteristic'] = clusters

# Split the sensor data into features and labels
X = sensor_data.drop('Gait Characteristic', axis=1).values
y = sensor_data['Gait Characteristic'].values

# Preprocess the data (scaling or normalization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data to fit the LSTM input shape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the RNN model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

@app.route('/')
def index():
    return render_template('gait.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    # Get the input data from the form
    input_data = []
    for key in request.form:
        input_data.append(float(request.form[key]))

    # Preprocess the input data
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

    # Perform predictions on the input data
    predictions = model.predict(input_data)
    predicted_class_index = np.argmax(predictions[0])

    # Map prediction labels to detailed recommendations
    detailed_recommendations = {
        0: 'Normal Gait: Your gait is within the normal range. Maintain good posture, stride length, and cadence.',
        1: 'Fast Gait: If you are aiming for a slower pace, focus on slowing down your walking speed. This can help improve stability and reduce the risk of tripping or falling.',
        2: 'Slow Gait: To increase your walking speed, focus on increasing your stride length and maintaining a steady pace. Regular exercise and strength training can also help improve your walking speed.',
        3: 'Limb Asymmetry: Work on improving symmetry between your limbs during walking. This may involve specific exercises or physical therapy to address any muscle imbalances or gait abnormalities.',
        4: 'Stiff Gait: To improve flexibility and reduce stiffness, incorporate stretching exercises into your daily routine. Warm-up exercises before walking can also help loosen up muscles and joints.',
        5: 'Ataxic Gait: Consult a healthcare professional for further evaluation and guidance on managing ataxic gait. They can provide specific exercises and interventions to address coordination and balance issues.',
        6: 'Antalgic Gait: Consult a healthcare professional for further evaluation and guidance on managing antalgic gait. They can help identify and address the underlying cause of the gait abnormality.',
        7: 'Trendelenburg Gait: Consult a healthcare professional for further evaluation and guidance on managing Trendelenburg gait. They can provide targeted exercises and interventions to address muscle weakness or imbalances in the hip and pelvis area.'
    }

    # Provide recommendations based on predictions
    recommendation = detailed_recommendations.get(predicted_class_index, 'Unable to provide a recommendation for the predicted gait characteristic.')

    return render_template('recommendation.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
