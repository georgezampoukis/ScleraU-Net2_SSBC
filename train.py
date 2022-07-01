import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import unet
import TrainFuncs
import MainFuncs

IMG_WIDTH = 560
IMG_HEIGHT = 320

batch_size = 8
epochs = 500
seed = 1

filters = 16
drop = 0.4

data_augmentation = False

Dataset = "MASD+SMD"    # 'ALL', 'MASD+SBVPI', 'MASD+SMD', 'SBVPI', 'SMD'


groups = 16

mode = 1    # 1: Train     2: Visualize Dataset     3: Predict and Display      4: Test

NAME = f"{Dataset}_unet_{IMG_WIDTH}x{IMG_HEIGHT}_{batch_size}_{filters}f_{drop}d_{data_augmentation}_{mode}_{groups}g_{int(time.time())}.h5"

saved_model = "MASD+SMD_unetGN_560x320_8_16f_0.4d_False_6_16g_1607004402.h5"

TRAIN_DATA_PATH = f"data/{Dataset}/train
VAL_DATA_PATH = f"data/{Dataset}/validation
TEST_DATA_PATH = f"data/{Dataset}/test


#   Train Mode
if mode == 1 or mode == 2:


    train_ids = next(os.walk(TRAIN_DATA_PATH + "/images/"))[2]
    val_ids = next(os.walk(VAL_DATA_PATH + "/images/"))[2]

    train_index = 0
    val_index = 0

    X = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    Y = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    X2 = np.zeros((len(val_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    Y2 = np.zeros((len(val_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    # LOAD AND PREPARE TRAIN IMAGES
    print("\nLoading {} Train Images".format(len(train_ids)))
    for n in tqdm(range(len(train_ids))):

        path_ = TRAIN_DATA_PATH
        while True:

            # LOAD IMAGES AND MASKS
            if os.path.exists(path_ + "/images/%d.png" % train_index):
                img = cv2.imread(path_ + "/images/%d.png" % train_index)
                mask = cv2.imread(path_ + "/masks/{}.png".format(train_index))
                train_index += 1
                break
            else:
                train_index += 1
                continue

        x_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        x_mask = x_mask.squeeze()

        x_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x_img = x_img.astype(np.float32)
        x_img = x_img / 255

        x_mask = cv2.resize(x_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x_mask = np.reshape(x_mask, (x_mask.shape[0], x_mask.shape[1], 1))
        x_mask = x_mask.astype(np.float32)
        x_mask = x_mask / 255

        X[n] = x_img
        Y[n] = x_mask

    # LOAD AND PREPARE VALIDATION IMAGES
    print("\nLoading {} Validation Images".format(len(val_ids)))
    for m in tqdm(range(len(val_ids))):

        path_ = VAL_DATA_PATH
        while True:

            # LOAD IMAGES AND MASKS
            if os.path.exists(path_ + "/images/%d.png" % val_index):
                img = cv2.imread(path_ + "/images/%d.png" % val_index)
                mask = cv2.imread(path_ + "/masks/{}.png".format(val_index))
                val_index += 1
                break
            else:
                val_index += 1
                continue

        x_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        x_mask = x_mask.squeeze()

        x_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x_img = x_img.astype(np.float32)
        x_img = x_img / 255

        x_mask = cv2.resize(x_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x_mask = np.reshape(x_mask, (x_mask.shape[0], x_mask.shape[1], 1))
        x_mask = x_mask.astype(np.float32)
        x_mask = x_mask / 255

        X2[m] = x_img
        Y2[m] = x_mask

    # SET THE FIT FUNCTION VALUES
    x_train = X
    y_train = Y
    x_val = X2
    y_val = Y2

    #   Visualize Mode
    if mode == 2:
        for i in range(len(x_train)):
            a = x_train[i]
            a = a.squeeze()
            a = a * 255
            a = a.astype(np.uint8)
            a = cv2.resize(a, (960, 540), interpolation=cv2.INTER_AREA)

            b = y_train[i]
            b = MainFuncs.ProcessPredictedMask(b, 11, 960, 540, 100, False, 1)

            contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

            for c in contours:
                cv2.drawContours(a, [c], -1, (0, 255, 0), 2)

            cv2.imshow("result", np.hstack((a, b)))
            cv2.waitKey(500)

    #   Train Mode
    if mode == 1:

        model = unet.unet(input_img=(IMG_HEIGHT, IMG_WIDTH, 3), n_filters=filters, dropout=drop)

        earlystopping = EarlyStopping(patience=10, verbose=1, mode='min', monitor='val_loss')

        modelcheckpoint = ModelCheckpoint(NAME, verbose=1, save_best_only=True, save_weights_only=True, mode='min', monitor='val_loss')


        callbacks = [earlystopping, modelcheckpoint]

        if data_augmentation:

            train_generator = TrainFuncs.Augmentor(x_data=x_train, y_data=y_train, batch_size=batch_size, seed=seed)
            val_generator = TrainFuncs.Augmentor(x_data=x_val, y_data=y_val, batch_size=batch_size, seed=seed)

            results = model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=len(x_train)/batch_size, 
                                            verbose=1, validation_steps=len(x_val)/batch_size, epochs=epochs, callbacks=callbacks)

        else:

            results = model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks)

#   Predict and Display Mode
if mode == 3:

    model = unet.unet(input_img=(IMG_HEIGHT, IMG_WIDTH, 3), n_filters=filters, dropout=drop)
    model.load_weights(saved_model)

    img = cv2.imread("361.png")
    frame = img

    img = MainFuncs.ProcessFrameForPrediction(img, IMG_WIDTH, IMG_HEIGHT)

    img_prediction = model.predict(img, verbose=1)

    pi = img_prediction
    pi = MainFuncs.ProcessPredictedMask(pi, 11, 960, 540, 100, True, 1)
    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)

    contours, _ = cv2.findContours(pi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pi = cv2.cvtColor(pi, cv2.COLOR_GRAY2BGR)

    for c in contours:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)

    cv2.imshow("result", np.hstack((frame, pi)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   Test Mode
if mode == 4:

    test_ids = next(os.walk(TEST_DATA_PATH + "/images/"))[2]

    X3 = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    Y3 = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    test_index = 0

    for k in tqdm(range(len(test_ids))):

        path_ = TEST_DATA_PATH

        while True:

            # LOAD IMAGES AND MASKS
            if os.path.exists(path_ + "/images/%d.png" % test_index):
                img = cv2.imread(path_ + "/images/%d.png" % test_index)
                mask = cv2.imread(path_ + "/masks/{}.png".format(test_index))
                test_index += 1
                break
            else:
                test_index += 1
                continue

        x_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        x_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x_img = x_img.astype(np.float32)
        x_img = x_img / 255

        x_mask = cv2.resize(x_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x_mask = np.reshape(x_mask, (x_mask.shape[0], x_mask.shape[1], 1))
        x_mask = x_mask.astype(np.float32)
        x_mask = x_mask / 255

        X3[k] = x_img
        Y3[k] = x_mask

    x_test = X3
    y_test = Y3

    model = unet.unet(input_img=(IMG_HEIGHT, IMG_WIDTH, 3), n_filters=filters, dropout=drop)
    model.load_weights(saved_model)
    evaluation = model.evaluate(x_test, y_test, batch_size=4)

    evaluation = "Dice Loss: %.5f, Mean IOU: %.5f" % (evaluation[0], evaluation[1])

    print(evaluation)
