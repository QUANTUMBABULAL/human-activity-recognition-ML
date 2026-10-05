"""Multi-layer perceptron (Keras). OWNER: Person 2.

Paper: 512-512 ReLU, dropout 0.5, softmax, categorical cross-entropy, SGD; 100 epochs shown for
the reduced set. The report gives no learning rate, momentum or batch size: we use the Keras
defaults of its time (SGD lr 0.01, no momentum, batch 32). Inputs are standardised.

KerasMLP wraps everything the runner must not see:
    original IDs {1..24} -> 0..11 (LABEL_ORDER) -> one-hot -> softmax -> argmax -> original IDs
    StandardScaler fitted in fit() on the fitting rows only; predict() only transforms.
    optional validation slice carved (seeded) from the rows passed to fit() - i.e. training
    rows only; it is used to monitor / early-stop, never the test split.

TensorFlow is imported lazily so `import har.models_p2.mlp` works without it (tests auto-skip).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from har.config import LABEL_ORDER, SEED

PAPER = {"hidden_units": (512, 512), "activation": "relu", "dropout": 0.5,
         "output_activation": "softmax", "loss": "categorical_crossentropy", "optimizer": "SGD",
         "epochs": 100}
# Report-text values (fractions). Reduced: Figure 2 shows 0.840, text and poster say 0.814 -> text.
PAPER_TEST_ACC = {"full": 0.981, "reduced": 0.814}
DEFAULTS = {"learning_rate": 0.01, "momentum": 0.0, "batch_size": 32, "epochs": 100,
            "val_frac": 0.1, "patience": None}
CLASSES = np.array(LABEL_ORDER, dtype=np.int64)


def encode_labels(y) -> np.ndarray:
    """Original activity IDs -> 0..11 by position in LABEL_ORDER. Unknown IDs raise."""
    y = np.asarray(y, dtype=np.int64)
    idx = np.searchsorted(CLASSES, y)
    bad = (idx >= len(CLASSES)) | (CLASSES[np.minimum(idx, len(CLASSES) - 1)] != y)
    if bad.any():
        raise ValueError(f"labels not in LABEL_ORDER: {sorted(set(y[bad].tolist()))}")
    return idx.astype(np.int64)


def decode_labels(idx) -> np.ndarray:
    """0..11 -> original activity IDs."""
    return CLASSES[np.asarray(idx, dtype=np.int64)]


def configure_reproducibility(seed: int = SEED, deterministic_ops: bool = True):
    """Seed Python, NumPy and TensorFlow (keras.utils.set_random_seed) and, optionally, force
    deterministic TF kernels (process-wide). Same machine + same versions -> same run on CPU;
    other hardware / oneDNN / thread counts may still differ in the last bits."""
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


class KerasMLP:
    """sklearn-style .fit/.predict wrapper around a 512-512 Keras MLP (see module docstring)."""

    def __init__(self, hidden_units=PAPER["hidden_units"], dropout: float = PAPER["dropout"],
                 learning_rate: float = DEFAULTS["learning_rate"], momentum: float = DEFAULTS["momentum"],
                 batch_size: int = DEFAULTS["batch_size"], epochs: int = DEFAULTS["epochs"],
                 val_frac: float = DEFAULTS["val_frac"], patience: int | None = DEFAULTS["patience"],
                 seed: int = SEED, deterministic_ops: bool = True, verbose: int = 0,
                 predict_batch_size: int = 8192):
        self.hidden_units, self.dropout = tuple(hidden_units), dropout
        self.learning_rate, self.momentum = learning_rate, momentum
        self.batch_size, self.epochs, self.val_frac, self.patience = batch_size, epochs, val_frac, patience
        self.seed, self.deterministic_ops, self.verbose = seed, deterministic_ops, verbose
        self.predict_batch_size = predict_batch_size
        self.classes_ = CLASSES

    def get_config(self) -> dict:
        return {"hidden_units": list(self.hidden_units), "activation": PAPER["activation"],
                "dropout": self.dropout, "output_units": len(self.classes_),
                "output_activation": PAPER["output_activation"], "loss": PAPER["loss"],
                "optimizer": "SGD", "learning_rate": self.learning_rate, "momentum": self.momentum,
                "batch_size": self.batch_size, "epochs": self.epochs, "val_frac": self.val_frac,
                "patience": self.patience, "scaler": "StandardScaler (fit on fitting rows only)",
                "seed": self.seed, "deterministic_ops": self.deterministic_ops}

    def build_network(self, n_features: int):
        import keras
        layers = [keras.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers += [keras.layers.Dense(units, activation=PAPER["activation"]),
                       keras.layers.Dropout(self.dropout, seed=self.seed)]
        layers.append(keras.layers.Dense(len(self.classes_), activation=PAPER["output_activation"]))
        net = keras.Sequential(layers)
        net.compile(optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate,
                                                   momentum=self.momentum),
                    loss=PAPER["loss"], metrics=["accuracy"])
        return net

    def _split_validation(self, n: int):
        """Seeded (fit, val) positions inside the rows given to fit(); val is empty if val_frac == 0."""
        n_val = int(round(self.val_frac * n))
        if n_val == 0:
            return np.arange(n), np.arange(0)
        perm = np.random.default_rng(self.seed).permutation(n)
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])

    def fit(self, X, y):
        import keras
        configure_reproducibility(self.seed, self.deterministic_ops)
        X = np.asarray(X, dtype=np.float32)
        y_idx = encode_labels(y)
        fit_pos, val_pos = self._split_validation(len(X))

        self.scaler_ = StandardScaler().fit(X[fit_pos])          # fitting rows only
        self.n_fit_rows_, self.n_val_rows_ = int(len(fit_pos)), int(len(val_pos))
        Xs = self.scaler_.transform(X).astype(np.float32)
        Y = keras.utils.to_categorical(y_idx, num_classes=len(self.classes_))

        self.model_ = self.build_network(X.shape[1])
        callbacks, val = [], None
        if len(val_pos):
            val = (Xs[val_pos], Y[val_pos])
            if self.patience:
                callbacks.append(keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience, restore_best_weights=True))
        hist = self.model_.fit(Xs[fit_pos], Y[fit_pos], batch_size=self.batch_size, epochs=self.epochs,
                               validation_data=val, shuffle=True, callbacks=callbacks,
                               verbose=self.verbose)
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        self.epochs_run_ = len(self.history_["loss"])
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)  # no refit
        return self.model_.predict(Xs, batch_size=self.predict_batch_size, verbose=0)

    def predict(self, X) -> np.ndarray:
        return decode_labels(np.argmax(self.predict_proba(X), axis=1))


def build(**hyperparams) -> KerasMLP:
    return KerasMLP(**hyperparams)


def mlp_info(model: KerasMLP) -> dict:
    """Training facts merged into the result JSON via run_experiment(post_fit=...)."""
    import keras
    h = model.history_
    info = {"epochs_run": model.epochs_run_, "n_fit_rows": model.n_fit_rows_,
            "n_val_rows": model.n_val_rows_, "n_params": int(model.model_.count_params()),
            "keras_version": keras.__version__,
            "final_train_loss": h["loss"][-1], "final_train_accuracy_epoch": h["accuracy"][-1]}
    if "val_loss" in h:
        best = int(np.argmin(h["val_loss"]))
        info.update(final_val_loss=h["val_loss"][-1], final_val_accuracy=h["val_accuracy"][-1],
                    best_val_loss_epoch=best + 1, best_val_accuracy=float(max(h["val_accuracy"])))
    return info
