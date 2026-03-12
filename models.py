from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, layers, callbacks
from tensorflow.data import Dataset



class MLP(keras.Model):
    def __init__(self, input_dim, hidden_layer_sizes=(512, 256, 128), output_dim=1, output_activation="sigmoid"):
        super().__init__()
        self.hidden_layers = []
        self.input_layer = layers.InputLayer(input_shape=(input_dim,))
        for units in hidden_layer_sizes:
            self.hidden_layers.append(layers.Dense(units, activation="elu"))
        self.output_layer = layers.Dense(output_dim, activation=output_activation)

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def fit(self, X_train, A_train, y_train,
            loss: losses.Loss,
            optimizer: optimizers.Optimizer,
            X_val=None, A_val=None, y_val=None,
            epochs=10,
            batch_size=1024,
            callbacks: List[callbacks.Callback]=None,
            verbose=1):
        # Use compiled loss and optimizer
        if callbacks is None:
            callbacks = []
        
        dataset = tf.data.Dataset.from_tensor_slices((X_train, A_train, y_train)).batch(batch_size)
        val_dataset = None
        if X_val is not None and y_val is not None and A_val is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, A_val, y_val)).batch(batch_size)
        
        history = {"loss": [], "val_loss": []}
        
        # Call on_train_begin for all callbacks
        for callback in callbacks:
            callback.on_train_begin()
        
        for epoch in range(epochs):
            # Call on_epoch_begin for all callbacks
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            
            epoch_loss = []
            for x_batch, a_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    y_pred = self(x_batch, training=True)
                    batch_loss = loss(y_batch, y_pred, a_batch)
                grads = tape.gradient(batch_loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                epoch_loss.append(batch_loss.numpy())
            history["loss"].append(float(tf.reduce_mean(epoch_loss)))
            
            # Validation
            if val_dataset is not None:
                val_losses = []
                for x_val, a_val, y_val_ in val_dataset:
                    y_pred_val = self(x_val, training=False)
                    val_loss = loss(y_val_, y_pred_val, a_val)
                    val_losses.append(val_loss.numpy())
                history["val_loss"].append(float(tf.reduce_mean(val_losses)))
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {history['loss'][-1]:.4f}" + (f", val_loss: {history['val_loss'][-1]:.4f}" if val_dataset is not None else ""))
            
            # Call on_epoch_end for all callbacks
            logs = {"loss": history["loss"][-1]}
            if val_dataset is not None:
                logs["val_loss"] = history["val_loss"][-1]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
        
        # Call on_train_end for all callbacks
        for callback in callbacks:
            callback.on_train_end()
        
        return history


    def evaluate(self, X_test, A_test, y_test, batch_size=1024, metrics=None, verbose=1):
        # Use compiled metrics if not provided
        dataset = tf.data.Dataset.from_tensor_slices((X_test, A_test, y_test)).batch(batch_size)
        metrics_to_use = metrics if metrics is not None else self.compiled_metrics
        results = {m.__name__: [] for m in metrics_to_use}
        for x_batch, a_batch, y_batch in dataset:
            y_pred = self(x_batch, training=False)
            for m in metrics_to_use:
                results[m.__name__].append(m(y_batch, y_pred, a_batch).numpy())
        summary = {k: float(tf.reduce_mean(v)) for k, v in results.items()}
        if verbose:
            print("Evaluation:", summary)
        return summary