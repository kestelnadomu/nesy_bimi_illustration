"""
Trainer for a tabular MLP in TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore")

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
        model,
        callbacks = None,
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

        self.model = model
        self.callbacks = callbacks
        self.history = None

    def _prepare_inputs(self, X_num, X_cat=None):
        inputs = {"numerical": X_num}
        if X_cat is not None:
            inputs["categorical"] = X_cat
        return inputs

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        X_num: np.ndarray,
        y: np.ndarray,
        X_cat: Optional[np.ndarray] = None,
        X_num_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_cat_val: Optional[np.ndarray] = None,
    ):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # ── Preprocessing ────────────────────────
        # self.preprocessor = TabularPreprocessor(
        #     numerical_cols=list(range(X_num.shape[1])),
        #     categorical_cols={},
        # )
        # X_num     = self.preprocessor.fit_transform_numerical(X_num)
        # X_num_val = self.preprocessor.transform_numerical(X_num_val)

        # # ── Build model ──────────────────────────
        # self.model = build_tabular_mlp(
        #     num_numerical=X_num.shape[1],
        #     cardinalities=self.cardinalities,
        #     hidden_dims=self.hidden_dims,
        #     dropout_rate=self.dropout_rate,
        #     output_units=self.output_units,
        #     task=self.task,
        #     use_skip_connections=self.use_skip_connections,
        # )

        # # ── LR Schedule ──────────────────────────
        # if self.use_cosine_schedule:
        #     steps_per_epoch = int(np.ceil(len(X_num) / self.batch_size))
        #     total_steps     = steps_per_epoch * self.max_epochs
        #     warmup_steps    = steps_per_epoch * 2
        #     lr = cosine_annealing_schedule(
        #         initial_lr=self.learning_rate,
        #         total_steps=total_steps,
        #         warmup_steps=warmup_steps,
        #     )
        # else:
        #     lr = self.learning_rate

        # optimizer = keras.optimizers.AdamW(
        #     learning_rate=lr,
        #     weight_decay=self.weight_decay,
        # )

        # self.model.compile(
        #     optimizer=optimizer,
        #     loss=get_loss(self.task, self.label_smoothing),
        #     metrics=get_metrics(self.task),
        # )

        # ── Train ────────────────────────────────

        
        train_inputs = self._prepare_inputs(X_num, X_cat)
        val_inputs   = self._prepare_inputs(X_num_val, X_cat_val)

        # callbacks = get_callbacks(
        #     checkpoint_path=self.checkpoint_path,
        #     patience=15,
        # )
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
