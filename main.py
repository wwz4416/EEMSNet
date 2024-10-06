import os
os.chdir("/root/siton-tmp/fct/")
from model import *
import numpy as np
import pdb
import h5py


# import tensorflow.compat.v1 as tf
# import tensorflow.compat.v1.keras.backend as K
# tf.disable_eager_execution()
#
# def get_flops(model):
#     run_meta = tf.RunMetadata()
#     opts = tf.profiler.ProfileOptionBuilder.float_operation()
#     flops = tf.profiler.profile(graph=K.get_session().graph,
#     run_meta=run_meta, cmd='op', options=opts)
#     return flops.total_float_ops

# %% Get data ACDC

# training
acdc_data, train_afn, train_hdr = get_acdc(acdc_data_train)
X_train, y_train, info_train = acdc_data[0], acdc_data[1], acdc_data[2]
# validation
acdc_data, val_afn, val_hdr = get_acdc(acdc_data_validation)
X_val, y_val, info_val = acdc_data[0], acdc_data[1], acdc_data[2]
# testing
acdc_data, test_afn, test_hdr = get_acdc(acdc_data_test)
X_test, y_test, info_test = acdc_data[0], acdc_data[1], acdc_data[2]

y_train_cat = convert_masks(y_train)
y_val_cat = convert_masks(y_val)
y_test_cat = convert_masks(y_test)

edg_train = edg_generate(y_train_cat)
edg_val = edg_generate(y_val_cat)
edg_test = edg_generate(y_test_cat)

batch_size = 2

train_generator = unite_gen(X_train, y_train_cat[:, ::4, ::4, :], y_train_cat[:, ::2, ::2, :], y_train_cat, edg_train,
                            batch_size, "training")
val_generator = unite_gen(X_val, y_val_cat[:, ::4, ::4, :], y_val_cat[:, ::2, ::2, :], y_val_cat, edg_val, batch_size,
                          "validation")

# %% Training

model = FCT(X_train)

# delete old logs
dirpath = Path("/root/siton-tmp/fct/myunet_tflogs")
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

name = "myUNet".format(time.strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard("myunet_tflogs/{}".format(name))

warmup_epoch = 8
warmup_run_epochs = 120
normal_run_epochs = 38

rlrop = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=5,
    factor=.5,
    # min_lr=1e-6,
    min_delta=.001,
    verbose=1)

checkpoint_filepath = "/root/siton-tmp/fct/myunet_weights/myunet_weights.h5"

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True
)


earlystop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    restore_best_weights=True,
    min_delta=.001,
    patience=12)

lr = 1e-3
opt = tf.keras.optimizers.Adam(lr=lr)

# Compute the number of warmup batches
warmup_batches = warmup_epoch * len(X_train) // batch_size
# Create the Learning rate scheduler
warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=lr)
#
loss1 = "binary_crossentropy"
model.compile(optimizer=opt,
              loss=[loss1, loss1, loss1, dice_loss],
              loss_weights=[.14, .29, .57, .02],
              )

# first training with warmup
model.fit(train_generator,
          steps_per_epoch=len(X_train) // batch_size,
          epochs=warmup_run_epochs,
          callbacks=[warm_up_lr],
          )

# second training with rlrop
history = model.fit(train_generator,
                    validation_data=val_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    validation_steps=len(X_val) // batch_size,
                    epochs=normal_run_epochs,
                    callbacks=[rlrop, checkpoint_callback, tensorboard_callback],
                    )

np.save('acdc_history_rlrop_bestmodel.npy', history.history)

# %%

predicted = model.predict(X_test, batch_size=1)

# only predicted[2] is of interest as this is the higest resolution output of Deep Supervision
print(np.round(np.array(metrics(y_test[:, :, :, -1], np.argmax(predicted[2], axis=3), 0)), 4))

# the average is:
print(np.round(np.array(metrics(y_test[:, :, :, -1], np.argmax(predicted[2], axis=3), 0)).mean(), 4))

#############################################################################

predicted = model.predict(X_val, batch_size=1)

# only predicted[2] is of interest as this is the higest resolution output of Deep Supervision
print(np.round(np.array(metrics(y_val[:, :, :, -1], np.argmax(predicted[2], axis=3), 0)), 4))

# the average is:
print(np.round(np.array(metrics(y_val[:, :, :, -1], np.argmax(predicted[2], axis=3), 0)).mean(), 4))