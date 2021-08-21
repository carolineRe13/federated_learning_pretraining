from autoencoder import Autoencoder

LATENT_DIMS = [10 * 10, 9 * 9, 8 * 8, 7 * 7, 6 * 6, 5 * 5, 4 * 4, 3 * 3]


def run_autoencoder_training():
    results = []

    for latent_dim in LATENT_DIMS:
        autoencoder = Autoencoder(latent_dim)
        results.append(autoencoder.train(None))

    # plot results with labels


if __name__ == '__main__':
    run_autoencoder_training()
