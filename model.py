"""The stacked LSTM regression model (same architecture as v1)."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(timesteps, num_features, learning_rate=0.001):
    model = keras.Sequential([
        keras.Input(shape=(timesteps, num_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
