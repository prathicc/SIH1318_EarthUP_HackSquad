from flask import Flask, render_template, request
import numpy as np
import numpy as np
import tensorflow as tf
import io
import base64
from PIL import Image

app = Flask(__name__, static_folder='static')

# Load the saved model
model = tf.keras.models.load_model("./lulc_20_epoch")
class_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/index3')
def index3():
    return render_template('index3.html')

@app.route('/index4')
def index4():
    return render_template('index4.html')

@app.route('/index5', methods=['POST', 'GET'])
def index5():
    if 'file' not in request.files:
        return render_template('index5.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index5.html', message='No selected file')
    print('Hello')
    # Read the image content
    img_content = file.read()

    # Open the image using PIL
    img = Image.open(io.BytesIO(img_content))
    # Resize the image to a specified width and height
    new_width, new_height = 128, 128  # Replace with your desired dimensions
    
    # Perform image preprocessing
    img = img.resize((64, 64))  # Resize if needed
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the corresponding class name
    predicted_class_name = class_labels[predicted_class_index]

    # Format the prediction result
    result = f"Prediction: {predicted_class_name}"
    img = img.resize((new_width, new_height))
    # Convert the image to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_content).decode('utf-8')
    img_data_uri = f'data:image/jpeg;base64,{img_base64}'
    print(predicted_class_name)
    # Format the prediction result
    result = {
        'prediction': predicted_class_name,
        'img_data_uri': img_data_uri
    }

    return render_template('index5.html', result=result)

if __name__ == '__main__': 
    app.run(debug=True)
