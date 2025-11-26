import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- PHASE 1: DATA PREPROCESSING ---

print("Setting up Image Generators...")

# 1. Training Data Generator (With Augmentation)
# We twist, zoom, and flip the training images so the AI learns better.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# 2. Test Data Generator (No Augmentation, just scaling)
test_datagen = ImageDataGenerator(rescale = 1./255)

# 3. Load the Images from folders
print("Loading Training Data...")
# Make sure your folder structure is D:\Xray_Project\chest_xray\train
training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

print("Loading Test Data...")
test_set = test_datagen.flow_from_directory('chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# --- PHASE 2: BUILDING THE CNN ---

print("Building CNN...")

cnn = tf.keras.models.Sequential()

# Step 1: Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4: Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5: Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print("CNN Architecture built successfully!")

# --- PHASE 3: TRAINING ---

# Compile
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train
print("Starting training... (This may take 5-10 minutes)")
# We save the history so we can plot it later if we want
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

print("Training Complete!")

# Save the model
cnn.save('xray_model.keras')
print("Model saved as 'xray_model.keras'")