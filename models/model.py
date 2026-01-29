"""
CNN Model Architectures for Product Fraud Detection System
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import config


def create_siamese_base_network(input_shape=(*config.IMAGE_SIZE, 3)):
    """
    Create the base CNN network for Siamese architecture
    This network will be used twice with shared weights
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Keras Model representing the base network
    """
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    
    # L2 normalization for better similarity computation
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    
    return Model(inputs, outputs, name='siamese_base')


def euclidean_distance(vectors):
    """
    Compute Euclidean distance between two vectors
    
    Args:
        vectors: List of two tensors [vector1, vector2]
        
    Returns:
        Euclidean distance
    """
    vector1, vector2 = vectors
    sum_squared = tf.reduce_sum(tf.square(vector1 - vector2), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))


def create_siamese_model(input_shape=(*config.IMAGE_SIZE, 3)):
    """
    Create complete Siamese network for product authentication
    
    Args:
        input_shape: Shape of input images
        
    Returns:
        Compiled Siamese model
    """
    # Create base network
    base_network = create_siamese_base_network(input_shape)
    
    # Define inputs for the two images
    input_a = keras.Input(shape=input_shape, name='original_product')
    input_b = keras.Input(shape=input_shape, name='returned_product')
    
    # Process both inputs through the same base network (shared weights)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Compute distance between the two embeddings
    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
    
    # Convert distance to similarity score (0 to 1)
    # Smaller distance = higher similarity
    similarity = layers.Lambda(lambda x: 1 / (1 + x))(distance)
    
    # Create the final model
    model = Model(inputs=[input_a, input_b], outputs=similarity, name='siamese_network')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def create_damage_model(input_shape=(*config.IMAGE_SIZE, 3), num_classes=config.NUM_DAMAGE_CLASSES):
    """
    Create CNN model for damage detection
    Uses transfer learning with MobileNetV2 as base
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of damage categories
        
    Returns:
        Compiled damage detection model
    """
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create new model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='damage_classification')(x)
    
    model = Model(inputs, outputs, name='damage_detector')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    return model


def unfreeze_base_model(model, num_layers_to_unfreeze=20):
    """
    Unfreeze the last N layers of the base model for fine-tuning
    
    Args:
        model: Damage detection model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    """
    base_model = model.layers[2]  # MobileNetV2 is the 3rd layer
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers_to_unfreeze
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Creating Siamese Network...")
    siamese_model = create_siamese_model()
    siamese_model.summary()
    print(f"\nSiamese model created successfully!")
    print(f"Total parameters: {siamese_model.count_params():,}")
    
    print("\n" + "="*80 + "\n")
    
    print("Creating Damage Detection Model...")
    damage_model = create_damage_model()
    damage_model.summary()
    print(f"\nDamage model created successfully!")
    print(f"Total parameters: {damage_model.count_params():,}")
