import datetime
from threading import Thread, Lock

from keras import backend as K
from keras.models import clone_model, Model
from keras.layers import Input, Dense, Lambda
from keras.callbacks import TensorBoard
import tensorflow as tf

from config.model import TENSORBOARD_LOG_DIR
from config.model import VAE_MODEL


LOCK = Lock()
latent_dim = 3
epochs = 1

class VAE:
    def __init__(self, x_shape, save_interval=100):
        """
        Initialize VAE setting
        :param x_shape: X shape(not x(i) shape)
        """

        m, n = x_shape
        hidden_unit_size = n >> 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.example = tf.placeholder(shape=(None, n), dtype=tf.float32)
            self.queue = tf.FIFOQueue(capacity=20, dtypes=[tf.float32])
            self.enqueue = self.queue.enqueue((self.example, ))
            self.qr = tf.train.QueueRunner(self.queue, [self.enqueue] * 4)
            self.coord = tf.train.Coordinator()

            # x = Input(shape=(n, ), name='x')
            x = Input(shape=(n, ), dtype=tf.float32, tensor=self.queue.dequeue(), name='x')
            h1 = Dense(hidden_unit_size, activation='relu', dtype=tf.float32, name='h1')(x)
            mean = Dense(latent_dim, name='mean')(h1)
            var = Dense(latent_dim, name='var')(h1)

            def sampling(args):
                z_mean, z_var = args
                epsilon = K.random_normal(shape=K.shape(z_var))
                return z_mean + z_var * epsilon
                # return z_mean + K.exp(z_var / 2) * epsilon

            z = Lambda(sampling, name='z')([mean, var])

            decoder_h1 = Dense(hidden_unit_size, activation='relu', name='decoder_h1')(z)
            y = Dense(n, activation='sigmoid', name='y')(decoder_h1)

            def loss(y_true, y_pred):
                kld = (-1 / 2) * (K.sum(1 + K.log(K.square(var)) - K.square(mean) - K.square(var), axis=1))
                # kld = (-1 / 2) * K.sum(1 + var - K.square(mean) - K.exp(var))
                re = K.mean(K.sum(K.binary_crossentropy(y_true, y_pred), axis=1))
                return K.mean(kld + re)

            model = Model(inputs=x, outputs=y)
            model.compile(optimizer='adam', loss=loss)

            # using learn
            self._model = model
            # using predict without being affected by learning
            self.model = clone_model(self._model)

            self.y = y
            e_x = Input(shape=(n, ), name='e_x')
            e_h1 = Dense(hidden_unit_size, activation='relu', name='e_h1')(e_x)
            e_mean = Dense(latent_dim, name='e_mean')(e_h1)
            e_var = Dense(latent_dim, name='e_var')(e_h1)
            e_z = Lambda(sampling, name='e_z')([e_mean, e_var])

            self.encoder = Model(inputs=e_x, outputs=e_z)

            z_input = Input(shape=(latent_dim,))
            d_h1 = Dense(hidden_unit_size, activation='relu', name='d_h1')(z_input)
            d_y = Dense(n, activation='sigmoid', name='d_y')(d_h1)

            self.decoder = Model(inputs=z_input, outputs=d_y)
            # self.a = tf.placeholder(dtype=tf.float32, shape=(None, 2))
            # self.b = tf.placeholder(dtype=tf.float32, shape=(None, 2))
            # self.ab = self.a + self.b

        self.session = tf.Session(graph=self.graph)
        K.set_session(self.session)

    def learn(self, x_train, x_test=None):
        if x_test is not None:
            validation_data = (x_test, x_test)
        else:
            validation_data = None

        enqueue_threads = self.qr.create_threads(self.session, coord=self.coord, start=True)

        with LOCK:
            for i in range(1):
                self.session.run(self.enqueue, feed_dict={self.example: x_train})

        self.coord.join(enqueue_threads)
            # with tf.Session(graph=K.get_session().graph):
            # self._model.fit(x=x_train, y=x_train, epochs=epochs, validation_data=validation_data,
            #                 callbacks=[TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)])

        with LOCK:
            w = self._model.get_weights()
            self.model.set_weights(w)
            self.encoder.set_weights(w[0:len(w) - 4])
            self.decoder.set_weights(w[-4:])

            self.model.save(VAE_MODEL + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.h5')

    def predict(self, x):
        return self.decoder.predict(self.encoder.predict(x))

    def encode(self, x):
        # with K.get_session() as sess:
        return self.encoder.predict(x)

    def decode(self, z):
        # with K.get_session() as sess:
        return self.decoder.predict(z)

    def _show_predict_image(self, x):
        import matplotlib.pyplot as plt
        import numpy as np
        pred = self.predict(x)
        plt.imshow(np.reshape(x[0], (28, 28)), cmap='Greys_r')
        plt.show()
        plt.imshow(np.reshape(pred[0], (28, 28)), cmap='Greys_r')
        plt.show()
        plt.imshow(np.reshape(x[5000], (28, 28)), cmap='Greys_r')
        plt.show()
        plt.imshow(np.reshape(pred[5000], (28, 28)), cmap='Greys_r')
        plt.show()

def _main(args):
    x_train, x_test = args
    vae = VAE(x_shape=x_train.shape)

    for _ in range(2):
        thread = Thread(target=vae.learn, kwargs={'x_train': x_train, 'x_test': x_test})
        thread.start()
    # vae.learn(x_train, x_test)
    # vae.learn(x_train, x_test)
    # print(thread.is_alive())
    # thread.join()
    # print(thread.is_alive())
    # vae._show_predict_image(x_test)

if __name__ == '__main__':
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    _main((x_train, x_test))
