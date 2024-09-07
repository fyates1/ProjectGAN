import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from loss_functions import generator_loss, discriminator_loss, identity_loss, calc_cycle_loss

class LinearDecayLR(Callback):
    def __init__(self, initial_lr, final_lr, decay_start_epoch, total_epochs, optimizers):
        super(LinearDecayLR, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_start_epoch = decay_start_epoch
        self.total_epochs = total_epochs
        self.optimizers = optimizers

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.decay_start_epoch:
            # Calculate the linear decay factor
            decay_epochs = self.total_epochs - self.decay_start_epoch
            lr_decay_rate = (self.initial_lr - self.final_lr) / decay_epochs
            new_lr = max(self.initial_lr - lr_decay_rate * (epoch - self.decay_start_epoch), self.final_lr)
        else:
            new_lr = self.initial_lr

        for optimizer in self.optimizers:
            tf.keras.backend.set_value(optimizer.lr, new_lr)

        # Output the current learning rate for each optimizer
        print(f"Epoch {epoch + 1}/{self.total_epochs}")
        for i, optimizer in enumerate(self.optimizers):
            current_lr = tf.keras.backend.get_value(optimizer.lr)
            print(f"Optimizer {i + 1} learning rate: {current_lr:.8f}")


@tf.function
def train_step(real_x, real_y, generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.

    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])

        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
