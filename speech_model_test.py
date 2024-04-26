import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential 
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.regularizers import l1_l2
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
from data_loader import load_data
import tf2onnx
import onnx

# Load your data
X_train, X_test, y_train, y_test = load_data()

def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        kernel_regularizer=l1_l2(l1=hp.Float('l1', 1e-6, 1e-2, sampling='log'),
                                 l2=hp.Float('l2', 1e-6, 1e-2, sampling='log')))))
    model.add(Dropout(rate=hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', 16, 128, step=16), activation='silu'))
    model.add(Dense(y_train.shape[-1], activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate the tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=2,
    directory='tuner_results',
    project_name='speech_recognition_tuning'
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Execute the search
tuner.search(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

# Build the model with the optimal hyperparameters and train it on the data for 20 epochs
model = build_model(best_hps)
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)

# Evaluate the model
eval_result = model.evaluate(X_test, y_test)
print(f"[Test Loss, Test Accuracy]: {eval_result}")

# Save the model in multiple formats in case of compatibility issues and for future-proofing

# Save the model in Keras format
model.save('speech_recognition_model.keras')

# Convert the model to ONNX forma
model.output_names = ['output']
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='digit')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, 'speech_recognition_model.onnx')

# # Save as Keras HDF5 model
# model.save('speech_recognition_model.h5')

# # Save the model in TensorFlow's SavedModel format
# model.save('/speech_recognition_model', save_format='tf')

# # Save model weights
# model.save_weights('speech_recognition_model_weights.h5')


print("Model saved.")
