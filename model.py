# model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from preprocessing import preprocess_signal

class_map = {'Normal': 0, 'Bearing Fault': 1, 'Misalignment': 2, 'Rotor Imbalance': 3}

# Custom layer to wrap tf.image.resize
class ResizeLayer(layers.Layer):
    def __init__(self, target_size, method='bilinear', **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size
        self.method = method

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method=self.method)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({'target_size': self.target_size, 'method': self.method})
        return config

def build_hybrid_model(num_classes=4):
    inputs = layers.Input(shape=(224, 224, 3))
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    for layer in resnet_base.layers[:143]:
        layer.trainable = False
    resnet_output = resnet_base.output
    
    transition = layers.Conv2D(1280, (1, 1), padding='same', activation='relu')(resnet_output)
    # Use the custom ResizeLayer instead of tf.image.resize directly
    upsampled = ResizeLayer(target_size=[224, 224], method='bilinear')(transition)
    upsampled = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(upsampled)
    
    efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in efficientnet_base.layers[:80]:
        layer.trainable = False
    for layer in efficientnet_base.layers[80:]:
        layer.trainable = True
    efficientnet_output = efficientnet_base(upsampled)
    efficientnet_output = layers.Dropout(0.5)(efficientnet_output)
    
    x = layers.GlobalAveragePooling2D()(efficientnet_output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.Dropout(0.7)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def train_and_evaluate(model, X, y, epochs=30, batch_size=16):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30, width_shift_range=0.3, height_shift_range=0.3,
        zoom_range=0.3, shear_range=0.3, horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        preprocessing_function=lambda x: x + np.random.normal(0, 0.05, x.shape)
    )
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                   lr_scheduler]
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nPer-Class Metrics:")
    for i, cls in enumerate(class_map.keys()):
        print(f"{cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # --- Plot Training Accuracy & Loss ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axs[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axs[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    axs[0].set_title("Model Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid()

    # Loss plot
    axs[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axs[1].plot(history.history['val_loss'], label='Val Loss', marker='o')
    axs[1].set_title("Model Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid()

    # Save plots
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/accuracy_loss_plot.png')
    plt.show()

    # --- Plot Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_map.keys(), yticklabels=class_map.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig('plots/confusion_matrix.png')
    plt.show()

    # --- Save the Model ---
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "hybrid_model"))  # SavedModel format
    model.save(os.path.join(save_dir, "hybrid_model.h5"))  # HDF5 format
    model.save_weights(os.path.join(save_dir, "hybrid_model.weights.h5"))
    
    return model, history, X_test, y_test

def real_time_inference(model, signal, fs=10000):
    start_time = time.time()
    spectrogram = preprocess_signal(signal)
    pred = model.predict(spectrogram[np.newaxis, ...])
    class_idx = np.argmax(pred)
    inference_time = time.time() - start_time
    class_names = list(class_map.keys())
    print(f"Predicted: {class_names[class_idx]}, Confidence: {pred[0][class_idx]:.2f}, Time: {inference_time:.3f}s")
    return class_idx, inference_time
