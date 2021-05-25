"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MSE


class LossLKB(keras.losses.Loss):
    def __init__(self, hyp_params, **kwargs):
        super(LossLKB, self).__init__(**kwargs)

        # Parameters
        self.a1 = hyp_params['a1']
        self.a2 = hyp_params['a2']
        self.a3 = hyp_params['a3']
        self.a4 = hyp_params['a4']
        self.a5 = hyp_params['a5']
        self.num_pred_steps = hyp_params['num_pred_steps']
        self.precision = hyp_params['precision']
        self.pretrain = hyp_params['pretrain']

        # Loss components
        self.loss_recon = tf.constant(0.0, dtype=self.precision)
        self.loss_lin = tf.constant(0.0, dtype=self.precision)
        self.loss_pred = tf.constant(0.0, dtype=self.precision)
        self.loss_inf = tf.constant(0.0, dtype=self.precision)
        self.loss_reg = tf.constant(0.0, dtype=self.precision)
        self.total_loss = tf.constant(0.0, dtype=self.precision)

    def call(self, model, obs):
        """ model = [y, x_ae, x_adv, y_adv, wnorms]

            model[0] : encoded time series (y)
                shape - (batch size, time steps, latent dim)
            model[1] : encoded-decoded time series (x_ae)
                shape - (batch size, time steps, phys dim)
            model[2] : ecoded-advanced-decoded time series (x_adv)
                shape - (batch size, time steps, phys dim)
            model[3] : encoded-advanced time series (y_adv)
                shape - (batch size, pred steps, latent dim)
            model[4] : normalized weights

            obs : The original input data
        Args:
            model:
            obs:

        Returns:

        """
        y = tf.identity(model[0])
        x_ae = tf.identity(model[1])
        x_adv = tf.identity(model[2])
        y_adv = tf.identity(model[3])
        weights = model[4]

        if self.pretrain:
            # Autoencoder reconstruction
            x_ae = tf.identity(model[1])
            self.loss_recon = tf.reduce_mean(MSE(obs, x_ae))
            self.total_loss = self.a1*self.loss_recon
        else:
            # Autoencoder reconstruction (only initial condition)
            self.loss_recon = tf.reduce_mean(MSE(obs[:, 0, :], x_ae[:, 0, :]))

            # Future state prediction
            self.loss_pred = tf.reduce_mean(MSE(obs[:, :self.num_pred_steps, :], x_adv[:, :self.num_pred_steps, :]))

            # Linearity of steps in the latent space
            self.loss_lin = tf.reduce_mean(MSE(y[:, :self.num_pred_steps, :], y_adv[:, :self.num_pred_steps, :]))

            # L-inf penalty
            self.loss_inf = tf.reduce_max(tf.abs(obs[:, 0, :] - x_ae[:, 0, :])) + \
                            tf.reduce_max(tf.abs(obs[:, 1, :] - x_adv[:, 1, :]))

            # Regularization on weights
            self.loss_reg = tf.add_n([tf.nn.l2_loss(w) for w in weights])

            # Total loss
            self.total_loss = self.a1 * self.loss_recon + self.a2 * self.loss_pred + self.a3 * self.loss_lin + \
                              self.a4 * self.loss_inf + self.a5 * self.loss_reg

        return self.total_loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'a1': self.a1,
                'a2': self.a2,
                'a3': self.a3,
                'num_pred_steps': self.num_pred_steps,
                'loss_recon': self.a1 * self.loss_recon,
                'loss_pred': self.a2 * self.loss_pred,
                'loss_lin': self.a3 * self.loss_lin,
                'loss_inf': self.a4 * self.loss_inf,
                'loss_reg': self.a5 * self.loss_reg,
                'total_loss': self.total_loss}
