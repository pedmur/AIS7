# Model definition (simplified & regularized)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

input_shape = x_train.shape[1:]  # (n_mels, time_steps, 1)

# 1. DEFINE THE ARCHITECTURE
# CRITICAL: This must match the original training code EXACTLY.
# If you used the "Improved Model" from our earlier chat, it looks like this:
def build_custom_model():
    model = Sequential([
        # Block 1
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128, 259, 1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        
        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
    
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    return model

# 2. CREATE THE EMPTY SHELL
model = build_custom_model()

# 3. LOAD THE WEIGHTS
# The numbers from the PC will fill the empty layers here.
try:
    model.load_weights("yvette_custom_weights.h5")
    print("Custom model weights loaded successfully!")
except ValueError as e:
    print("Shape Mismatch! Did you change the number of filters or layers?")
    print("The architecture in this script must match Yvette's training script exactly.")
    print(e)
    
# Use a smaller learning rate to stabilise training
model.compile(
    optimizer=Adam(learning_rate=1e-1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Now you are ready to predict
# prediction = model.predict(input_data)