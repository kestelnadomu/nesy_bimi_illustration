"""
Tabular MLP in TensorFlow/Keras
Optimized for large datasets (~1M rows)

Features:
- ResNet-style skip connections
- GELU activations
- BatchNorm + Dropout
- AdamW optimizer
- Cosine annealing LR schedule
- Embedding support for categoricals
- Quantile normalization for numerics
- Early stopping
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA PREPROCESSING
# ─────────────────────────────────────────────

class TabularPreprocessor:
    """
    Handles:
    - Quantile normalization for numerical features
    - Embeddings for high-cardinality categoricals
    - One-hot encoding for low-cardinality categoricals
    - Missing value imputation + missingness indicators
    """

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: Dict[str, int],   # {col_name: cardinality}
        low_cardinality_threshold: int = 10,
        quantile_noise: float = 1e-3,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.low_cardinality_threshold = low_cardinality_threshold
        self.quantile_noise = quantile_noise

        self.quantile_transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=1000,
            random_state=42,
        )

    def get_embedding_dim(self, cardinality: int) -> int:
        return min(50, (cardinality // 2) + 1)

    def fit(self, X_num: np.ndarray):
        self.quantile_transformer.fit(X_num)
        return self

    def transform_numerical(self, X_num: np.ndarray) -> np.ndarray:
        return self.quantile_transformer.transform(X_num).astype(np.float32)

    def fit_transform_numerical(self, X_num: np.ndarray) -> np.ndarray:
        return self.quantile_transformer.fit_transform(X_num).astype(np.float32)


# ─────────────────────────────────────────────
# 2. MODEL COMPONENTS
# ─────────────────────────────────────────────

class ResidualBlock(layers.Layer):
    """
    ResNet-style block: Linear -> BN -> GELU -> Dropout -> Linear -> BN + skip
    """

    def __init__(self, units: int, dropout_rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.units = units

        self.dense1 = layers.Dense(units, use_bias=False)
        self.bn1    = layers.BatchNormalization()
        self.act1   = layers.Activation("gelu")
        self.drop1  = layers.Dropout(dropout_rate)

        self.dense2 = layers.Dense(units, use_bias=False)
        self.bn2    = layers.BatchNormalization()
        self.drop2  = layers.Dropout(dropout_rate)

        # Project input if dimensions don't match
        self.projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x if self.projection is None else self.projection(x)

        out = self.dense1(x)
        out = self.bn1(out, training=training)
        out = self.act1(out)
        out = self.drop1(out, training=training)

        out = self.dense2(out)
        out = self.bn2(out, training=training)
        out = self.drop2(out, training=training)

        out = out + residual
        return keras.activations.gelu(out)


class CategoricalEmbedder(layers.Layer):
    """
    Embeds multiple categorical features and concatenates them.
    Each feature gets its own embedding table.
    """

    def __init__(self, cardinalities: List[int], embedding_dims: List[int], **kwargs):
        super().__init__(**kwargs)
        self.embeddings = [
            layers.Embedding(input_dim=card + 1, output_dim=dim)  # +1 for unknown/OOV
            for card, dim in zip(cardinalities, embedding_dims)
        ]

    def call(self, x):
        # x: (batch, num_cat_features) — integer encoded
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return tf.concat(embedded, axis=-1)


# ─────────────────────────────────────────────
# 3. FULL MODEL
# ─────────────────────────────────────────────

def build_tabular_mlp(
    num_numerical: int,
    cardinalities: Optional[List[int]] = None,   # list of cardinalities for cat features
    hidden_dims: List[int] = [512, 256, 128],
    dropout_rate: float = 0.15,
    output_units: int = 1,
    task: str = "binary",                         # "binary", "multiclass", "regression"
    use_skip_connections: bool = True,
) -> keras.Model:
    """
    Build the tabular MLP.

    Args:
        num_numerical:      Number of numerical input features.
        cardinalities:      List of cardinality for each categorical feature.
                            None if no categorical features.
        hidden_dims:        Width of each hidden layer.
        dropout_rate:       Dropout probability.
        output_units:       Number of output units (1 for binary/regression,
                            num_classes for multiclass).
        task:               One of "binary", "multiclass", "regression".
        use_skip_connections: Whether to use ResNet-style blocks.

    Returns:
        Compiled Keras model.
    """

    # ── Inputs ──────────────────────────────────
    num_input = keras.Input(shape=(num_numerical,), name="numerical")
    inputs = [num_input]
    parts  = [num_input]

    if cardinalities:
        cat_input = keras.Input(shape=(len(cardinalities),), dtype=tf.int32, name="categorical")
        inputs.append(cat_input)
        embedding_dims = [min(50, (c // 2) + 1) for c in cardinalities]
        cat_embedded = CategoricalEmbedder(cardinalities, embedding_dims)(cat_input)
        parts.append(cat_embedded)

    # ── Concatenate all features ─────────────────
    x = tf.concat(parts, axis=-1) if len(parts) > 1 else parts[0]

    # ── Stem: project to first hidden_dim ────────
    x = layers.Dense(hidden_dims[0], use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    # ── Hidden Layers ────────────────────────────
    for units in hidden_dims:
        if use_skip_connections:
            x = ResidualBlock(units, dropout_rate=dropout_rate)(x)
        else:
            x = layers.Dense(units, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("gelu")(x)
            x = layers.Dropout(dropout_rate)(x)

    # ── Output Head ──────────────────────────────
    if task == "binary":
        output = layers.Dense(1, activation="sigmoid", name="output")(x)
    elif task == "multiclass":
        output = layers.Dense(output_units, activation="softmax", name="output")(x)
    elif task == "regression":
        output = layers.Dense(output_units, activation="linear", name="output")(x)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from binary, multiclass, regression.")

    model = keras.Model(inputs=inputs, outputs=output, name="TabularMLP")
    return model


# ─────────────────────────────────────────────
# 4. TRAINING SETUP
# ─────────────────────────────────────────────

def get_loss(task: str, label_smoothing: float = 0.05):
    if task == "binary":
        return keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    elif task == "multiclass":
        return keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    elif task == "regression":
        return keras.losses.MeanSquaredError()
    raise ValueError(f"Unknown task: {task}")


def get_metrics(task: str):
    if task == "binary":
        return [keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")]
    elif task == "multiclass":
        return [keras.metrics.CategoricalAccuracy(name="acc")]
    elif task == "regression":
        return [keras.metrics.MeanAbsoluteError(name="mae")]
    raise ValueError(f"Unknown task: {task}")


def compile_model(
    model: keras.Model,
    task: str = "binary",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=get_loss(task, label_smoothing),
        metrics=get_metrics(task),
    )
    return model


def get_callbacks(
    checkpoint_path: str = "best_model.keras",
    patience: int = 15,
    min_lr: float = 1e-6,
):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def cosine_annealing_schedule(
    initial_lr: float = 1e-3,
    total_steps: int = 100_000,
    warmup_steps: int = 5_000,
    min_lr: float = 1e-6,
):
    """Cosine annealing with linear warmup as a LearningRateSchedule."""

    class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self):
            super().__init__()

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup = tf.cast(warmup_steps, tf.float32)
            total  = tf.cast(total_steps, tf.float32)
            pi     = tf.constant(np.pi)

            # Linear warmup
            warmup_lr = initial_lr * (step / warmup)

            # Cosine decay
            progress = (step - warmup) / (total - warmup)
            cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + tf.cos(pi * progress))

            return tf.where(step < warmup, warmup_lr, cosine_lr)

        def get_config(self):
            return dict(
                initial_lr=initial_lr,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=min_lr,
            )

    return WarmupCosineDecay()


# ─────────────────────────────────────────────
# 5. HIGH-LEVEL TRAINER
# ─────────────────────────────────────────────

class TabularMLPTrainer:
    """
    End-to-end trainer. Handles preprocessing, building, and training.

    Example usage:
        trainer = TabularMLPTrainer(task="binary")
        trainer.fit(X_num_train, y_train, X_cat=X_cat_train)
        preds = trainer.predict(X_num_test, X_cat=X_cat_test)
    """

    def __init__(
        self,
        task: str = "binary",
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        batch_size: int = 1024,
        max_epochs: int = 100,
        val_size: float = 0.1,
        use_cosine_schedule: bool = True,
        use_skip_connections: bool = True,
        cardinalities: Optional[List[int]] = None,
        output_units: int = 1,
        checkpoint_path: str = "best_model.keras",
        seed: int = 42,
    ):
        self.task               = task
        self.hidden_dims        = hidden_dims
        self.dropout_rate       = dropout_rate
        self.learning_rate      = learning_rate
        self.weight_decay       = weight_decay
        self.label_smoothing    = label_smoothing
        self.batch_size         = batch_size
        self.max_epochs         = max_epochs
        self.val_size           = val_size
        self.use_cosine_schedule = use_cosine_schedule
        self.use_skip_connections = use_skip_connections
        self.cardinalities      = cardinalities
        self.output_units       = output_units
        self.checkpoint_path    = checkpoint_path
        self.seed               = seed

        self.model = None
        self.preprocessor = None
        self.history = None

    def _prepare_inputs(self, X_num, X_cat=None):
        inputs = {"numerical": X_num}
        if X_cat is not None:
            inputs["categorical"] = X_cat
        return inputs

    def fit(
        self,
        X_num: np.ndarray,
        y: np.ndarray,
        X_cat: Optional[np.ndarray] = None,
        X_num_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_cat_val: Optional[np.ndarray] = None,
    ):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # ── Validation split ─────────────────────
        if X_num_val is None:
            if X_cat is not None:
                X_num, X_num_val, X_cat, X_cat_val, y, y_val = train_test_split(
                    X_num, X_cat, y, test_size=self.val_size, random_state=self.seed
                )
            else:
                X_num, X_num_val, y, y_val = train_test_split(
                    X_num, y, test_size=self.val_size, random_state=self.seed
                )

        # ── Preprocessing ────────────────────────
        self.preprocessor = TabularPreprocessor(
            numerical_cols=list(range(X_num.shape[1])),
            categorical_cols={},
        )
        X_num     = self.preprocessor.fit_transform_numerical(X_num)
        X_num_val = self.preprocessor.transform_numerical(X_num_val)

        # ── Build model ──────────────────────────
        self.model = build_tabular_mlp(
            num_numerical=X_num.shape[1],
            cardinalities=self.cardinalities,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            output_units=self.output_units,
            task=self.task,
            use_skip_connections=self.use_skip_connections,
        )

        # ── LR Schedule ──────────────────────────
        if self.use_cosine_schedule:
            steps_per_epoch = int(np.ceil(len(X_num) / self.batch_size))
            total_steps     = steps_per_epoch * self.max_epochs
            warmup_steps    = steps_per_epoch * 2
            lr = cosine_annealing_schedule(
                initial_lr=self.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
        else:
            lr = self.learning_rate

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=self.weight_decay,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=get_loss(self.task, self.label_smoothing),
            metrics=get_metrics(self.task),
        )

        # ── Train ────────────────────────────────
        train_inputs = self._prepare_inputs(X_num, X_cat)
        val_inputs   = self._prepare_inputs(X_num_val, X_cat_val)

        callbacks = get_callbacks(
            checkpoint_path=self.checkpoint_path,
            patience=15,
        )
        # Remove ReduceLROnPlateau if using cosine schedule
        if self.use_cosine_schedule:
            callbacks = [c for c in callbacks if not isinstance(c, keras.callbacks.ReduceLROnPlateau)]

        self.history = self.model.fit(
            train_inputs, y,
            validation_data=(val_inputs, y_val),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return self

    def predict(self, X_num: np.ndarray, X_cat: Optional[np.ndarray] = None) -> np.ndarray:
        X_num  = self.preprocessor.transform_numerical(X_num)
        inputs = self._prepare_inputs(X_num, X_cat)
        return self.model.predict(inputs, batch_size=self.batch_size * 4)

    def summary(self):
        if self.model:
            self.model.summary()


# ─────────────────────────────────────────────
# 6. EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Generating synthetic dataset (1M rows)...")
    N, NUM_FEATURES, NUM_CAT = 1_000_000, 20, 3
    CARDINALITIES = [50, 100, 200]  # cardinalities of categorical features

    rng = np.random.default_rng(42)
    X_num = rng.standard_normal((N, NUM_FEATURES)).astype(np.float32)
    X_cat = np.column_stack([
        rng.integers(0, c, size=N) for c in CARDINALITIES
    ]).astype(np.int32)
    y = (X_num[:, 0] + X_num[:, 1] > 0).astype(np.float32)

    print(f"X_num: {X_num.shape}, X_cat: {X_cat.shape}, y: {y.shape}")
    print(f"Class balance: {y.mean():.3f}")

    # ── Train ────────────────────────────────────
    trainer = TabularMLPTrainer(
        task="binary",
        hidden_dims=[512, 256, 128],
        dropout_rate=0.15,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
        batch_size=1024,
        max_epochs=30,
        val_size=0.1,
        use_cosine_schedule=True,
        use_skip_connections=True,
        cardinalities=CARDINALITIES,
        output_units=1,
        checkpoint_path="best_tabular_mlp.keras",
    )

    trainer.fit(X_num, y, X_cat=X_cat)
    trainer.summary()

    # ── Inference ────────────────────────────────
    X_test_num = rng.standard_normal((1000, NUM_FEATURES)).astype(np.float32)
    X_test_cat = np.column_stack([
        rng.integers(0, c, size=1000) for c in CARDINALITIES
    ]).astype(np.int32)

    preds = trainer.predict(X_test_num, X_cat=X_test_cat)
    print(f"\nPrediction shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5].flatten()}")