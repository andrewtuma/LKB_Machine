"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
import numpy as np
import pickle
import datetime as dt
from scipy.integrate import solve_ivp
import os

import sys
sys.path.insert(0, '../../')
import LKBMachine as dl
import LossLKB as lf
import Training as tr


# ==============================================================================
# Setup
# ==============================================================================
NUM_SAVES = 1       # Number of times to save the model throughout training
NUM_PLOTS = 20      # Number of diagnostic plots to generate while training
DEVICE = '/GPU:0'
GPUS = tf.config.experimental.list_physical_devices('GPU')
if GPUS:
    try:
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    DEVICE = '/CPU:0'

tf.keras.backend.set_floatx('float64')  # !! Set precision for the entire model here
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Num GPUs Available: {}".format(len(GPUS)))
print("Training at precision: {}".format(tf.keras.backend.floatx()))
print("Training on device: {}".format(DEVICE))



# ==============================================================================
# Initialize hyper-parameters and Koopman model
# ==============================================================================
# General parameters
hyp_params = dict()
hyp_params['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp_params['experiment'] = 'van_der_pol'
hyp_params['plot_path'] = './training_results/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']
hyp_params['model_path'] = './trained_models/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']
hyp_params['device'] = DEVICE
hyp_params['precision'] = tf.keras.backend.floatx()
hyp_params['num_init_conds'] = 15000
hyp_params['num_train_init_conds'] = 10000
hyp_params['num_val_init_conds'] = 3000
hyp_params['num_test_init_conds'] = 2000
hyp_params['time_final'] = 2
hyp_params['delta_t'] = 0.01
hyp_params['mu'] = 1.5
hyp_params['num_time_steps'] = int(hyp_params['time_final']/hyp_params['delta_t'])
hyp_params['num_pred_steps'] = hyp_params['num_time_steps']
hyp_params['max_epochs'] = 100
hyp_params['save_every'] = hyp_params['max_epochs'] // NUM_SAVES
hyp_params['plot_every'] = hyp_params['max_epochs'] // NUM_PLOTS
hyp_params['pretrain'] = True
hyp_params['num_pretrain'] = 10

# Universal network layer parameters (AE & Aux)
hyp_params['optimizer'] = 'adam'
hyp_params['batch_size'] = 256
hyp_params['phys_dim'] = 2
hyp_params['num_cmplx_prs'] = 1
hyp_params['num_real'] = 0
hyp_params['latent_dim'] = 2 * hyp_params['num_cmplx_prs'] + hyp_params['num_real']
hyp_params['hidden_activation'] = tf.keras.activations.relu
hyp_params['bias_initializer'] = tf.keras.initializers.Zeros

# Encoding/Decoding Layer Parameters
hyp_params['num_en_layers'] = 3
hyp_params['num_en_neurons'] = 64
hyp_params['kernel_init_enc'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['kernel_init_dec'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['ae_output_activation'] = tf.keras.activations.linear

# Auxiliary Layer Parameters
hyp_params['num_k_layers'] = 1
hyp_params['num_k_neurons'] = 128
hyp_params['kernel_init_aux'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['aux_output_activation'] = tf.keras.activations.linear

# Loss Function Parameters
hyp_params['a1'] = tf.constant(1, dtype=hyp_params['precision'])  # Reconstruction
hyp_params['a2'] = tf.constant(1, dtype=hyp_params['precision'])  # Prediction
hyp_params['a3'] = tf.constant(1, dtype=hyp_params['precision'])  # Linearity
hyp_params['a4'] = tf.constant(1e-9, dtype=hyp_params['precision'])  # L-inf
hyp_params['a5'] = tf.constant(1e-14, dtype=hyp_params['precision'])  # L-2 on weights

# Learning rate
hyp_params['lr'] = 1e-3  # Learning rate

# Initialize the Koopman model and loss
myMachine = dl.LKBMachine(hyp_params)
myLoss = lf.LossLKB(hyp_params)


# ==============================================================================
# Generate / load data
# ==============================================================================
data_fname = 'vdp_data.pkl'
if os.path.exists(data_fname):
    # Load data from file
    data = pickle.load(open(data_fname, 'rb'))
    data = tf.cast(data, dtype=hyp_params['precision'])
else:
    def vdp(t, x):
        return [x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]]
    mu = hyp_params['mu']
    icx = np.random.uniform(-2, 2, hyp_params['num_init_conds'])
    icy = np.random.uniform(-2, 2, hyp_params['num_init_conds'])
    tspan = np.array([0, hyp_params['time_final']])
    dts = np.arange(0, hyp_params['time_final'], hyp_params['delta_t'])
    X = np.zeros(shape=(hyp_params['num_init_conds'], 2, hyp_params['num_time_steps']))
    for ii, ic in enumerate(zip(icx, icy)):
        tmp = solve_ivp(vdp, t_span=tspan, y0=ic, method='RK45', t_eval=dts)
        X[ii, :, :] = tmp.y
    data = tf.transpose(X, perm=[0, 2, 1])
    data = tf.cast(data, dtype=hyp_params['precision'])
    pickle.dump(data, open(data_fname, 'wb'))

# Normalize
dat = data.numpy()
x1min, x1max, x1mean = np.min(dat[:, :, 0]), np.max(dat[:, :, 0]), np.mean(dat[:, :, 0])
x2min, x2max, x2mean = np.min(dat[:, :, 1]), np.max(dat[:, :, 1]), np.mean(dat[:, :, 1])
dat[:, :, 0] = (dat[:, :, 0] - x1mean) / (x1max - x1min)
dat[:, :, 1] = (dat[:, :, 1] - x2mean) / (x2max - x2min)
data = dat

# Create training and validation datasets from the initial conditions
shuffled_data = tf.random.shuffle(data)
ntic = hyp_params['num_train_init_conds']
nvic = hyp_params['num_val_init_conds']
train_data = tf.data.Dataset.from_tensor_slices(shuffled_data[:ntic, :, :])
val_data = tf.data.Dataset.from_tensor_slices(shuffled_data[ntic:ntic+nvic, :, :])

# Batch and prefetch the validation data to the GPUs
val_set = val_data.batch(hyp_params['batch_size'], drop_remainder=True)
val_set = val_set.prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# Train the model
# ==============================================================================
results = tr.train_model(hyp_params=hyp_params, train_data=train_data,
                         val_set=val_set, model=myMachine, loss=myLoss)
print(results['model'].summary())
