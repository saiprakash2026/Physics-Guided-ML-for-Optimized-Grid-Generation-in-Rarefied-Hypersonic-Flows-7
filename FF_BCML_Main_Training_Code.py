# ==========================================================
# === IMPORTS ==============================================
# ==========================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split


# ==========================================================
# === HELPER FUNCTION ======================================
# ==========================================================

def iter_loadtxt(filename,
                 delimiter=',',
                 skiprows=0,
                 dtype=float,
                 usecols=None):

    print('File Reading Start')

    def iter_func():

        with open(filename, 'r') as infile:

            for _ in range(skiprows):
                next(infile)

            for line in infile:

                line = line.rstrip().split(delimiter)

                if usecols is not None:
                    selected = [line[i] for i in usecols]
                else:
                    selected = line

                for item in selected:
                    yield dtype(item)

        iter_loadtxt.rowlength = (
            len(usecols)
            if usecols is not None
            else len(line)
        )

    data = np.fromiter(iter_func(), dtype=dtype)

    data = data.reshape((-1, iter_loadtxt.rowlength))

    return data


# ==========================================================
# === NORMALIZATION CONSTANTS ==============================
# ==========================================================

VelocityNormalization = 12000.00
PressureNormalization = 21300.0
TemperatureNormalization = 65000.00

speed1 = temp1 = pres1 = 0


# ==========================================================
# === LOAD INPUT FILE ======================================
# ==========================================================

X = iter_loadtxt(
    "Input_All_NR.txt",
    delimiter=None,
    skiprows=0,
    dtype=float
)

for i in range(np.size(X[:, 0])):

    counter = 0
    speed = 0
    temp = 0

    for j in range(16):

        counter += 1

        for k in range(4):

            if k == 0:

                if np.abs(X[i, counter]) > speed:

                    speed = np.abs(X[i, counter])

                    pres = 100 * X[i, counter + 2]

                    temp = 300 * X[i, counter + 3]

            if k == 2:
                X[i, counter] *= 100

            if k == 3:
                X[i, counter] *= 300

            counter += 1

        for k in range(4):
            counter += 1

    if temp < 1.0:

        pres = pres1
        speed = speed1
        temp = temp1

    mach = speed / np.sqrt(1.4 * 287.0 * temp)

    p2 = pres * (
        1.0 + 2.0 * 1.4 * (mach ** 2 - 1.0) / 2.4
    )

    pres1, speed1, temp1 = pres, speed, temp

    counter = 0

    for j in range(16):

        counter += 1

        for k in range(4):
            counter += 1

        for k in range(4):

            if k == 2 and X[i, counter] == 1:
                X[i, counter] = p2

            counter += 1

print('File reading done')


# ==========================================================
# === EXPONENTIAL TRANSFORM ================================
# ==========================================================

for col in range(0, 136 + 1, 9):

    X[:, col] = np.exp((-X[:, col] ** 0.5))


# ==========================================================
# === NORMALIZE INPUT ======================================
# ==========================================================

for offset in range(0, 136 + 1, 9):

    X[:, offset + 1] /= VelocityNormalization
    X[:, offset + 2] /= VelocityNormalization
    X[:, offset + 3] /= PressureNormalization
    X[:, offset + 4] /= TemperatureNormalization


# pressure columns
for col in range(7, 143, 9):

    X[:, col] /= PressureNormalization


# ==========================================================
# === LOAD OUTPUTS =========================================
# ==========================================================

Y = np.genfromtxt("Output_AllNR.txt")

Y[:, 0] /= VelocityNormalization
Y[:, 1] /= VelocityNormalization
Y[:, 2] /= PressureNormalization
Y[:, 3] /= TemperatureNormalization

print('Normalization Done')


# ==========================================================
# === TRAIN-TEST SPLIT =====================================
# ==========================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train shape:", X_train.shape, Y_train.shape)
print("Test shape :", X_test.shape, Y_test.shape)


# ==========================================================
# === DEFINE MODEL =========================================
# ==========================================================

model = Sequential()

model.add(
    Dense(
        64,
        input_dim=X_train.shape[1],
        activation='relu'
    )
)

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(4, activation='linear'))

print(model.summary())


# ==========================================================
# === CUSTOM PROPERTY-WISE METRICS =========================
# ==========================================================

def mae_u(y_true, y_pred):

    return tf.reduce_mean(
        tf.abs(y_true[:, 0] - y_pred[:, 0])
    )


def mae_v(y_true, y_pred):

    return tf.reduce_mean(
        tf.abs(y_true[:, 1] - y_pred[:, 1])
    )


def mae_p(y_true, y_pred):

    return tf.reduce_mean(
        tf.abs(y_true[:, 2] - y_pred[:, 2])
    )


def mae_t(y_true, y_pred):

    return tf.reduce_mean(
        tf.abs(y_true[:, 3] - y_pred[:, 3])
    )


# ==========================================================
# === COMPILE MODEL ========================================
# ==========================================================

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[
        'mae',
        mae_u,
        mae_v,
        mae_p,
        mae_t
    ]
)


# ==========================================================
# === TRAIN MODEL ==========================================
# ==========================================================

nepoch = 200
nbatch = 32768

history = model.fit(
    X_train,
    Y_train,
    epochs=nepoch,
    batch_size=nbatch,
    validation_data=(X_test, Y_test),
    verbose=1
)


# ==========================================================
# === SAVE PROPERTY-WISE TRAINING HISTORY ==================
# ==========================================================

history_dict = pd.DataFrame({

    'epoch': np.arange(1, nepoch + 1),

    # overall loss
    'train_loss_mse': history.history['loss'],
    'val_loss_mse': history.history['val_loss'],

    # overall mae
    'train_mae': history.history['mae'],
    'val_mae': history.history['val_mae'],

    # velocity-u
    'train_mae_u': history.history['mae_u'],
    'val_mae_u': history.history['val_mae_u'],

    # velocity-v
    'train_mae_v': history.history['mae_v'],
    'val_mae_v': history.history['val_mae_v'],

    # pressure
    'train_mae_p': history.history['mae_p'],
    'val_mae_p': history.history['val_mae_p'],

    # temperature
    'train_mae_t': history.history['mae_t'],
    'val_mae_t': history.history['val_mae_t']

})

csv_filename = 'training_metrics_propertywise.csv'

history_dict.to_csv(csv_filename, index=False)

print(f"Saved property-wise training metrics to '{csv_filename}'")


# ==========================================================
# === EVALUATE MODEL =======================================
# ==========================================================

score = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)

print('Test loss (MSE):', score[0])
print('Test MAE:', score[1])


# ==========================================================
# === SAVE MODEL ===========================================
# ==========================================================

model.save('training_metrics_NR_1502A.keras')

print("Training completed successfully")
