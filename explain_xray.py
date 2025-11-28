import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# --- 1. SETUP ---
IMAGE_PATH = 'test_xray.jpeg'
MODEL_PATH = 'xray_model.keras'

print("🧠 Loading AI Radiologist...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- 2. ROBUST MODEL RECONSTRUCTION (The Fix) ---
# We rebuild the model functionally to avoid Keras 3 "Sequential" errors
print("🔧 Reconstructing model graph...")

# Find the last convolutional layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if 'conv2d' in layer.name:
        last_conv_layer_name = layer.name
        break

print(f"🔍 Found last convolutional layer: {last_conv_layer_name}")

# Create a new Input tensor
new_input = tf.keras.Input(shape=(64, 64, 3))

# Re-run the layers on this new input to capture the outputs
x = new_input
last_conv_output = None

for layer in model.layers:
    x = layer(x)
    if layer.name == last_conv_layer_name:
        last_conv_output = x

# Create the Grad-CAM model with explicit inputs/outputs
grad_model = tf.keras.models.Model(
    inputs=new_input,
    outputs=[last_conv_output, x]
)

# --- 3. PREPROCESS IMAGE ---
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0

img_array = get_img_array(IMAGE_PATH, size=(64, 64))

# --- 4. COMPUTE HEATMAP ---
print("🔥 Calculating Heatmap...")
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    class_channel = preds[:, 0]

grads = tape.gradient(class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# --- 5. VISUALIZATION ---
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (500, 500))

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img * 0.6
superimposed_img = np.uint8(superimposed_img)

print("✅ Done! Opening image...")
cv2.imshow('AI Logic Explanation (Press any key to exit)', superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()