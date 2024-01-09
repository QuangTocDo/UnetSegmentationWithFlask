from flask import Flask, render_template, request, jsonify  
import tensorflow as tf
import numpy as np
import os
import cv2
import base64
from PIL import Image

print(tf.__version__)
app = Flask(__name__)

def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

def save_result_image(result, result_path):
    # Assuming result is in the range [0, 1], multiply by 255 to bring it back to [0, 255]
    result *= 255
    result = result.astype(np.uint8)

    # Save the result image
    cv2.imwrite(result_path, result)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Process image data from the form
            image = request.files['image']
            image_path = os.path.join('./static/uploads', image.filename)  # Assuming 'uploads' is a folder to store the images
            image.save(image_path)

            # Resize the image if necessary
            img_array = preprocess_image(image_path)

            # Load the model
            model_path = os.path.abspath('./model')
            model = tf.keras.models.load_model(model_path)

            # Perform prediction on the input image
            result = model.predict(np.expand_dims(img_array, axis=0))

            # Ensure result and img_array have the same shape
            result_image = cv2.resize(result[0], (img_array.shape[1], img_array.shape[0]))

            # Save the result image
            result_image_path = os.path.join('./static/results', 'result_image.jpg')
            save_result_image(result_image, result_image_path)

            original_image_path = os.path.join('./static/uploads', image.filename)  # Assuming original image is in 'uploads'
            original_image = cv2.imread(original_image_path)
            result = cv2.imread(result_image_path)
            result = cv2.resize(result, (original_image.shape[1], original_image.shape[0]))
            # Apply alpha blending
            blended_image = cv2.addWeighted(original_image, 0.6, result, .8, 0)
            # Save the blended image
            blended_image_path = os.path.join('./static/results', 'blended_image.jpg')
            cv2.imwrite(blended_image_path, blended_image)

            # Convert the image from NumPy array to base64 format
            _, result_image = cv2.imencode('.jpg', blended_image)
            result_image = base64.b64encode(result_image.tobytes()).decode('utf-8')

            # Return the prediction result as JSON
            return jsonify(result_image=result_image)

        except Exception as e:
            print(e)
            return jsonify(error='Something went wrong. ' + str(e))
    
    return render_template('index.html', result_image=None, error=None)

if __name__ == '__main__':
    app.run(debug=True)
