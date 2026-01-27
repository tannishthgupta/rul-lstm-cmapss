import pandas as pd
import numpy as np

train_df=pd.read_csv("train_FD002.txt",sep=" ",header=None)




#print(train_df_1.shape)


train_df=train_df.dropna(axis=1)
#print(train_df_1.shape)

columns=(["engine_id","cycle"]+[f"op_setting_{i}" for i in range(1,4)]+[f"sensor_{i}" for i in range(1,22)])
train_df.columns=columns

#print(train_df_1.head())

max_cycles=train_df.groupby("engine_id")["cycle"].max()
train_df["max_cycle"]=train_df["engine_id"].map(max_cycles)
train_df["RUL"]=train_df["max_cycle"]-train_df["cycle"]

#print(train_df_1[["engine_id", "cycle", "max_cycle", "RUL"]].head(10))

train_df=train_df.drop(columns=["max_cycle"])

engine_1=train_df[train_df["engine_id"]==1]

import matplotlib.pyplot as plt 

"""plt.figure()
plt.plot(engine_1["cycle"],engine_1["RUL"])
plt.xlabel("cycle")
plt.ylabel("remaining useful life")
plt.title("cycle vs rul")
plt.show()"""



"""sensors_to_plot=["sensor_9","sensor_11","sensor_4","sensor_3"]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_sensors=scaler.fit_transform(engine_1[sensors_to_plot])
plt.figure()
for i,  sensor in enumerate(sensors_to_plot):
    plt.plot(engine_1["cycle"],scaled_sensors[:,i],label=sensor)

plt.xlabel("cycle")
plt.ylabel("normalized vals")
plt.title("cycle vs sensor data")
plt.legend()
plt.show()"""




sensor_cols=[col for col in train_df.columns if col.startswith("sensor")]
sensor_std=train_df[sensor_cols].std()

#print(sensor_std.sort_values(ascending=False).head(5))
#print(sensor_std.sort_values().head(5))

useful_sensor=sensor_std[sensor_std>0.01].index.tolist()
corr_with_rul=train_df[useful_sensor + ["RUL"]].corr()["RUL"]
corr_with_rul=corr_with_rul.drop("RUL")
selected_sensors=corr_with_rul[abs(corr_with_rul)>0.2].index.tolist()

final_features=selected_sensors + ["op_setting_1","op_setting_2","op_setting_3"]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
train_df[final_features]=scaler.fit_transform(train_df[final_features])

window_size=20

def create_sequences(x,y,window_size):
    x_seq=[]
    y_seq=[]
    for i in range(window_size,len(x)):
        x_seq.append(x[i-window_size:i])
        y_seq.append(y[i])

    return np.array(x_seq), np.array(y_seq)

x_all=[]
y_all=[]
for engine_id in train_df["engine_id"].unique():
    engine_data=train_df[train_df["engine_id"]==engine_id]

    x_engine=engine_data[final_features].values
    y_engine=engine_data["RUL"].values

    x_seq,y_seq=create_sequences(x_engine,y_engine,window_size)

    x_all.append(x_seq)
    y_all.append(y_seq)


x_all=np.concatenate(x_all)
y_all=np.concatenate(y_all)


print(type(x_all))
print(x_all.shape)
print(type(window_size), window_size)


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all,
    test_size=0.2,
    random_state=42
)

timesteps = int(x_train.shape[1])
num_features = int(x_train.shape[2])


 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model=Sequential()
model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(timesteps,num_features)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=32,
        return_sequences=False
    )
)

model.add(Dense(32,activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping 

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history=model.fit(
    x_train,y_train,
    validation_data=(x_val,y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1              
)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history["loss"], label="Train MSE")
plt.plot(history.history["val_loss"], label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.save("rul_lstm_fd002.keras")


test_df=pd.read_csv("test_FD002.txt", sep=" ", header=None)
test_df=test_df.dropna(axis=1)
test_df.columns=columns

true_rul = pd.read_csv("RUL_FD002.txt", header=None)
true_rul.columns = ["RUL"]

test_features=test_df[final_features]

test_features_scaled=scaler.transform(test_features)
test_df[final_features]=test_features_scaled


x_test=[]

for engine_id in test_df["engine_id"].unique():
    engine_data=test_df[test_df["engine_id"]==engine_id]

    if len(engine_data) >= window_size:
        seq = engine_data[final_features].values[-window_size:]
        x_test.append(seq)

x_test=np.array(x_test)

y_pred= model.predict(x_test).flatten()

from sklearn.metrics import mean_absolute_error , mean_squared_error

mae=mean_absolute_error(true_rul["RUL"],y_pred)
rmse=np.sqrt(mean_squared_error(true_rul["RUL"],y_pred))

print("Test MAE:", mae)
print("Test RMSE:", rmse)

plt.figure()
plt.scatter(true_rul["RUL"], y_pred, alpha=0.7)
plt.plot([0, max(true_rul["RUL"])], [0, max(true_rul["RUL"])], "r--")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("True vs Predicted RUL (FD002 Test)")
plt.show()
