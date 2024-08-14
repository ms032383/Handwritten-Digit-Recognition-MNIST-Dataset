from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('model/mnist_cnn_model.h5')


def preprocess_image(image):
    # Convert the image to grayscale, resize it to 28x28 pixels, and normalize it
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(img)

    # Predict the class of the digit
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction, axis=1)[0]

    return jsonify({'digit': int(predicted_digit)})


if __name__ == '__main__':
    app.run(debug=True)
