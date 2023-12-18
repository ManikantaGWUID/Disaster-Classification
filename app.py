import base64
from io import BytesIO
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
import pickle
import cv2
# Load pre-trained VGG19 model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Image Classification Web App"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        multiple=False
    ),
    html.Div(id='output-container')
])

# Define callback to handle image upload and prediction
@app.callback(
    Output('output-container', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        # Decode image from base64
        content_type, content_string = contents.split(',')
        decoded_image = base64.b64decode(content_string)
        image_np = np.frombuffer(decoded_image, dtype=np.uint8)
        image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Resize image to VGG19 input size
        image_cv2 = cv2.resize(image_cv2, (500, 500))
        # image_array = tf.keras.preprocessing.image.img_to_array(image_cv2)
        image_array = image_cv2/255
        image_array = image_array.reshape(1,500,500,3)
        mapper_output = {0:'Cyclone',  1:'Earthquake', 2:'Flood', 3:'NoRisk', 4:'Wildfire'}

        # Make prediction
        predictions = model.predict(image_array)[0]
        decoded_predictions = np.argsort(predictions)[::-1][:2]
        decode_predictions = {}
        for i in decoded_predictions:
            decode_predictions[mapper_output[i]] = predictions[i]
        
        # Display predictions
        result = html.Div([
            html.Img(src=contents),
            html.H4(f"Predictions for {filename}:"),
            html.H4(f"Predicted as {mapper_output[decoded_predictions[0]]}"),
            html.H4(f"Predicted probability {predictions[decoded_predictions[0]]*100}")
        ])
        return result
    else:
        return html.Div("Upload an image to make a prediction.")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
