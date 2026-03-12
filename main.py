from logging import config

import yaml
from fairml_datasets import Dataset
from tensorflow import keras


import models
import losses

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

def main():
    # ---------------------
    # Load configuration
    # ---------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)


    # ---------------------
    # Load data
    # ---------------------
    # Get the dataset
    dataset = Dataset.from_id(cfg["data"]["dataset_id"]) # [!code highlight]

    # Load as pandas DataFrame
    df = dataset.load()
    print(f"Dataset shape: {df.shape}")

    # Get the target column
    feature_cols = dataset.get_feature_columns()
    target_col = dataset.get_target_column()
    print(f"Feature columns: {feature_cols}")
    print(f"Target column: {target_col}")

    # Get sensitive attributes (before transformation)
    sensitive_columns = dataset.sensitive_columns

    if cfg["data"]["transform"]:
        # Transform to e.g. impute missing data
        df, transformation_info = dataset.transform(df)
        # Sensitive columns may change due to transformation
        sensitive_columns = transformation_info.sensitive_columns

    print(f"Sensitive columns: {sensitive_columns}")


    # ---------------------
    # Prepare data
    # ---------------------
    # Split into train, val, test sets
    train_df, val_df, test_df = dataset.train_test_val_split(df, test_size=cfg["data"]["test_size"], val_size=cfg["data"]["val_size"])
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    # Get indices of sensitive features
    sensitive_feature_idx = [i for i, col in enumerate(feature_cols) if col in sensitive_columns]

    # Prepare numpy arrays for model training
    X_train = train_df[feature_cols].values
    A_train = train_df[sensitive_columns].values
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].values
    A_val = val_df[sensitive_columns].values
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols].values
    A_test = test_df[sensitive_columns].values
    y_test = test_df[target_col].values

    # ---------------------
    # Train and evaluate models
    # ---------------------
    mlp = models.MLP

    # ── LR Schedule ──────────────────────────
    if cfg["train"]["cosine_schedule"]:
        steps_per_epoch = int(np.ceil(len(X_train) / cfg["train"]["batch_size"]))
        total_steps     = steps_per_epoch * cfg["train"]["epochs"]
        warmup_steps    = steps_per_epoch * 2
        lr = cosine_annealing_schedule( # TODO
            initial_lr=cfg["train"]["learning_rate"],
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
    else:
        lr = cfg["train"]["learning_rate"]

    opt = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=cfg["train"]["weight_decay"],
        )

    loss_experiment = {
        "baseline": losses.BaselineLoss(),
        "separation_variance": losses.SeparationLoss(
                type="variance",
                reg_weight=cfg["train"]["reg_weight"]
            ),
        "separation_symbolic": losses.SeparationLoss(
                type="symbolic",
                reg_weight=cfg["train"]["reg_weight"]
            ),
        "equaol_oppportunity_variance": losses.EqualOpportunityLoss(
                type="variance",
                reg_weight=cfg["train"]["reg_weight"]
            ),
        "equal_oppportunity_symbolic": losses.EqualOpportunityLoss(
                type="symbolic",
                reg_weight=cfg["train"]["reg_weight"]
            ),
    }

    for name, loss in loss_experiment.items():
        print(f"Training and evaluating model: {name}")
        model = mlp.copy()
        model.compile(
            optimizer = opt,
            loss = loss,
            metrics = ["accuracy"], # TODO
        )

        history = model.fit(
            X_train, A_train, y_train,
            loss = loss,
            optimizer = opt,
            validation_data = (X_val, A_val, y_val),
            epochs = cfg["train"]["epochs"],
            batch_size = cfg["train"]["batch_size"],
            callbacks = get_callbacks(
                    checkpoint_path=f"{name}_{cfg['train']['checkpoint_path']}",
                    patience=cfg["train"]["patience"],
                    min_lr=cfg["train"]["min_lr"],
                ),
            verbose = 1,
        )
        eval = model.evaluate(
            X_test, A_test, y_test,
            batch_size=cfg["train"]["batch_size"],
            verbose=1
        )
        print(f"Evaluation for {name}: {eval}")


if __name__ == "__main__":
    main()
