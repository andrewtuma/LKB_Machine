"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class LKBMachine(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(LKBMachine, self).__init__(**kwargs)

        # Parameters
        self.phys_dim = hyp_params['phys_dim']
        self.latent_dim = hyp_params['latent_dim']
        self.num_pred_steps = hyp_params['num_pred_steps']
        self.num_en_layers = hyp_params['num_en_layers']
        self.num_k_layers = hyp_params['num_k_layers']
        self.num_cmplx = hyp_params['num_cmplx_prs']
        self.num_real = hyp_params['num_real']
        self.mixed = self.num_real > 0 and self.num_cmplx > 0
        self.delta_t = hyp_params['delta_t']
        self.precision = hyp_params['precision']
        self.kernel_size = 1
        self.enc_input = (1, self.phys_dim)
        self.dec_input = (1, self.latent_dim)
        self.real_input = (1, self.num_real)
        self.cmplx_input = (1, self.num_cmplx)

        # Construct the ENCODER network
        self.encoder = keras.Sequential()
        self.encoder.add(Conv1D(filters=hyp_params['num_en_neurons'], kernel_size=self.kernel_size,
                                input_shape=self.enc_input,
                                activation=hyp_params['hidden_activation'],
                                kernel_initializer=hyp_params['kernel_init_enc'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='enc_in'))
        for ii in range(self.num_en_layers - 1):
            self.encoder.add(Conv1D(hyp_params['num_en_neurons'], kernel_size=self.kernel_size,
                                    activation=hyp_params['hidden_activation'],
                                    padding='same',
                                    kernel_initializer=hyp_params['kernel_init_enc'],
                                    bias_initializer=hyp_params['bias_initializer'],
                                    trainable=True, name='enc_' + str(ii)))
        self.encoder.add(Conv1D(self.latent_dim, kernel_size=self.kernel_size,
                                activation=hyp_params['ae_output_activation'],
                                padding='same',
                                kernel_initializer=hyp_params['kernel_init_enc'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='enc_out'))

        # Construct the DECODER network
        self.decoder = keras.Sequential(name="decoder")
        self.decoder.add(Conv1D(filters=hyp_params['num_en_neurons'], kernel_size=self.kernel_size,
                                input_shape=self.dec_input,
                                activation=hyp_params['hidden_activation'],
                                padding='same',
                                kernel_initializer=hyp_params['kernel_init_enc'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='dec_in'))
        for ii in range(self.num_en_layers - 1):
            self.decoder.add(Conv1D(hyp_params['num_en_neurons'], kernel_size=self.kernel_size,
                                    activation=hyp_params['hidden_activation'],
                                    padding='same',
                                    kernel_initializer=hyp_params['kernel_init_dec'],
                                    bias_initializer=hyp_params['bias_initializer'],
                                    trainable=True, name='dec_' + str(ii)))
        self.decoder.add(Conv1D(self.phys_dim, kernel_size=self.kernel_size,
                                activation=hyp_params['ae_output_activation'],
                                padding='same',
                                kernel_initializer=hyp_params['kernel_init_dec'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='dec_out'))

        # REAL eigenvalue auxiliary network
        if self.num_real > 0:
            self.auxnet_real = keras.Sequential(name="auxiliary_real")
            self.auxnet_real.add(Conv1D(filters=hyp_params['num_k_neurons'], kernel_size=self.kernel_size,
                                        input_shape=self.real_input,
                                        activation=hyp_params['hidden_activation'],
                                        padding='same',
                                        kernel_initializer=hyp_params['kernel_init_aux'],
                                        bias_initializer=hyp_params['bias_initializer'],
                                        trainable=True, name='aux_real_in'))
            for ii in range(self.num_k_layers - 1):
                self.auxnet_real.add(Conv1D(hyp_params['num_k_neurons'], kernel_size=self.kernel_size,
                                            activation=hyp_params['hidden_activation'],
                                            padding='same',
                                            kernel_initializer=hyp_params['kernel_init_aux'],
                                            bias_initializer=hyp_params['bias_initializer'],
                                            trainable=True, name='aux_real_' + str(ii)))
            self.auxnet_real.add(Conv1D(self.num_real, kernel_size=self.kernel_size,
                                        activation=hyp_params['aux_output_activation'],
                                        padding='same',
                                        kernel_initializer=hyp_params['kernel_init_aux'],
                                        bias_initializer=hyp_params['bias_initializer'],
                                        trainable=True, name='aux_real_out'))

        # COMPLEX eigenvalue auxiliary network
        if self.num_cmplx > 0:
            self.auxnet_cmplx = keras.Sequential(name="auxiliary_cmplx")
            self.auxnet_cmplx.add(Conv1D(filters=hyp_params['num_k_neurons'], kernel_size=self.kernel_size,
                                         input_shape=self.cmplx_input,
                                         activation=hyp_params['hidden_activation'],
                                         padding='same',
                                         kernel_initializer=hyp_params['kernel_init_aux'],
                                         bias_initializer=hyp_params['bias_initializer'],
                                         trainable=True, name='aux_cmplx_in'))
            for ii in range(self.num_k_layers - 1):
                self.auxnet_cmplx.add(Conv1D(hyp_params['num_k_neurons'], kernel_size=self.kernel_size,
                                             activation=hyp_params['hidden_activation'],
                                             padding='same',
                                             kernel_initializer=hyp_params['kernel_init_aux'],
                                             bias_initializer=hyp_params['bias_initializer'],
                                             trainable=True, name='aux_cmplx_' + str(ii)))
            self.auxnet_cmplx.add(Conv1D(2 * self.num_cmplx, kernel_size=self.kernel_size,
                                         activation=hyp_params['aux_output_activation'],
                                         padding='same',
                                         kernel_initializer=hyp_params['kernel_init_aux'],
                                         bias_initializer=hyp_params['bias_initializer'],
                                         trainable=True, name='aux_cmplx_out'))

    def call(self, x):
        # Encode the entire time series
        y = self.encoder(x)
        y0 = y[:, 0, :]
        y0 = y0[:, tf.newaxis, :]

        # Decode the entire time series
        x_ae = self.decoder(y)

        # Advance the trajectories
        evals_real = None
        evals_cmplx = None
        if self.mixed:
            y_adv, evals_cmplx, evals_real = self.advance_mixed(y0)
        elif self.num_real > 0:
            y_adv, evals_real = self.advance_real(y0)
        else:
            y_adv, evals_cmplx = self.advance_cmplx(y0)

        # Decode the latent trajectories
        x_adv = self.decoder(y_adv)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv, weights, evals_real, evals_cmplx]

    @tf.function
    def advance_real(self, y0):
        y_adv_tmp = tf.TensorArray(self.precision, size=self.num_pred_steps)
        evals_tmp = tf.TensorArray(self.precision, size=self.num_pred_steps)
        prior = y0
        y_adv_tmp = y_adv_tmp.write(0, prior)
        evals_tmp = evals_tmp.write(0, self.auxnet_real(prior))
        for tstep in tf.range(1, self.num_pred_steps):
            evals = self.auxnet_real(prior)
            evals_tmp = evals_tmp.write(tstep, evals)
            y_adv = tf.math.exp(evals * self.delta_t) * prior
            y_adv_tmp = y_adv_tmp.write(tstep, y_adv)
            prior = y_adv
        y_adv_tmp = tf.transpose(tf.squeeze(y_adv_tmp.stack()), perm=[1, 0, 2])
        evals_tmp = tf.transpose(tf.squeeze(evals_tmp.stack()), perm=[1, 0, 2])
        return y_adv_tmp, evals_tmp

    @tf.function
    def advance_cmplx(self, y0):
        y_adv_tmp = tf.TensorArray(self.precision, size=self.num_pred_steps)
        evals_tmp = tf.TensorArray(self.precision, size=self.num_pred_steps)
        prior = y0
        y_adv_tmp = y_adv_tmp.write(0, prior)
        radii = tf.math.add(tf.square(prior[:, :, 0::2]), tf.square(prior[:, :, 1::2]))
        evals_tmp = evals_tmp.write(0, self.auxnet_cmplx(radii))
        for tstep in tf.range(1, self.num_pred_steps):
            radii = tf.math.add(tf.square(prior[:, :, 0::2]), tf.square(prior[:, :, 1::2]))
            evals_cmplx = self.auxnet_cmplx(radii)
            evals_tmp = evals_tmp.write(tstep, evals_cmplx)
            mu = evals_cmplx[:, :, :self.num_cmplx]
            omega = evals_cmplx[:, :, self.num_cmplx:]
            e_mu = tf.math.exp(mu * self.delta_t)
            e_mu_cos_omega = tf.squeeze(e_mu * tf.math.cos(omega * self.delta_t), axis=1)
            e_mu_sin_omega = tf.squeeze(e_mu * tf.math.sin(omega * self.delta_t), axis=1)
            row1 = tf.stack([e_mu_cos_omega, -e_mu_sin_omega], axis=-1)
            row2 = tf.stack([e_mu_sin_omega, e_mu_cos_omega], axis=-1)
            K = tf.squeeze(tf.stack([row1, row2], axis=-1))
            y_adv = tf.linalg.matvec(K, tf.squeeze(prior, axis=1))
            y_adv = y_adv[:, tf.newaxis, :]
            y_adv_tmp = y_adv_tmp.write(tstep, y_adv)
            prior = y_adv
        y_adv_tmp = tf.transpose(tf.squeeze(y_adv_tmp.stack()), perm=[1, 0, 2])
        evals_tmp = tf.transpose(tf.squeeze(evals_tmp.stack()), perm=[1, 0, 2])
        return y_adv_tmp, evals_tmp

    @tf.function
    def advance_mixed(self, y0):
        tmp = tf.TensorArray(self.precision, size=self.num_pred_steps)
        zeros = tf.zeros((y0.shape[0], self.num_cmplx), dtype=self.precision)
        prior = y0
        tmp = tmp.write(0, prior)
        for tstep in tf.range(1, self.num_pred_steps):
            y_real = prior[:, :, :self.num_real]
            y_cmplx = prior[:, :, self.num_real:]

            # Real eigenvalues
            evals = self.auxnet_real(y_real)
            e_lam = tf.squeeze(tf.math.exp(evals * self.delta_t), axis=1)

            # Complex eigenvalues
            radii = tf.math.add(tf.square(y_cmplx[:, :, 0::2]), tf.square(y_cmplx[:, :, 1::2]))
            evals_cmplx = self.auxnet_cmplx(radii)
            mu = evals_cmplx[:, :, :self.num_cmplx]
            omega = evals_cmplx[:, :, self.num_cmplx:]
            e_mu = tf.math.exp(mu * self.delta_t)
            e_mu_cos_omega = tf.squeeze(e_mu * tf.math.cos(omega * self.delta_t), axis=1)
            e_mu_sin_omega = tf.squeeze(e_mu * tf.math.cos(omega * self.delta_t), axis=1)

            # Buid K matrix
            row1 = tf.stack([e_mu_cos_omega, -e_mu_sin_omega, zeros], axis=-1)
            row2 = tf.stack([e_mu_sin_omega, e_mu_cos_omega, zeros], axis=-1)
            row3 = tf.stack([zeros, zeros, e_lam], axis=-1)
            K = tf.squeeze(tf.stack([row1, row2, row3], axis=-1))

            # Advancing y coordinates
            y_adv = tf.linalg.matvec(K, tf.squeeze(prior, axis=1))
            y_adv = y_adv[:, tf.newaxis, :]
            tmp = tmp.write(tstep, y_adv)
            prior = y_adv
        return tf.transpose(tf.squeeze(tmp.stack()), perm=[1, 0, 2])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'num_pred_steps': self.num_pred_steps,
                'new_pred_depth': self.new_pred_depth,
                'autoencoder': self.autoencoder,
                'auxnet_real': self.auxnet_real}
