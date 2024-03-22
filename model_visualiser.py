# Description: Visualize the Keras model using visualkeras
import tensorflow as tf
from tensorflow import keras
import visualkeras
from PIL import ImageFont

# Load the Keras model
model = keras.models.load_model('speech_recognition_model.keras')
model.summary()

fontpath = 'fonts/Arial.ttf'  # custom font
font = ImageFont.truetype(fontpath, 10)

# Custom color palette for the layers
color_map = {
    keras.layers.InputLayer: 'green',  # Change 'green' to any color you like for Input layers
    keras.layers.Bidirectional: 'red',  # Change 'red' to any color you like for Bidirectional layers
    keras.layers.LSTM: 'blue',  # Change 'blue' to any color you like for LSTM layers
    keras.layers.Dropout: 'purple',  # Change 'purple' to any color you like for Dropout layers
    keras.layers.Dense: 'orange',  # Change 'orange' to any color you like for Dense layers
    # Add more layers and colors as needed
}

# Visualize the model
# visualkeras.layered_view(model, legend=True, to_file='model_visualization.png', color_map=color_map, font=font)
# visualkeras.layered_view(model, legend=True, to_file='model_visualization.png', font=font)
visualkeras.layered_view(model, 
                         legend=True, 
                         to_file='model_visualization.png', 
                         scale_xy=30,  # Adjusts the scale of the image
                         scale_z=30,  # Adjusts the scale of the image
                         max_z=65,  # Maximum number of layers to display
                         max_xy=10,  # Maximum number of layers to display
                         spacing=10,  # Adds more whitespace around the model
                         font=font)  # Font for the legend

