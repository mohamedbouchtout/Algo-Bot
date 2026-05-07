import numpy as np
from my_RBM_tf2_test import RBM
from bas_data import get_data
from utils import plot_image_grid, plot_input_sample
import os
import glob

rng = np.random.default_rng(seed=42)
raw = get_data(rng, s=4)

# Wrap into the dict format train() expects
split = int(len(raw) * 0.8)
data = {
    'x_train': np.array(raw[:split]),
    'x_test':  np.array(raw[split:])
}

print("x_train shape:", data['x_train'].shape)
print("x_test shape:", data['x_test'].shape)

rbm = RBM(
    visible_dim=16,
    hidden_dim=22,
    number_of_epochs=1000,
    picture_shape=(4, 4),
    batch_size=10,
    initial_temperature=1,
    annealing_decay=0,
    training_algorithm='cd',
    k=100,
    n_test_samples=6,
    NAME='test_run',
    initial_gamma=1.0, 
    gamma_decay=0.0
)

class SimpleOptimizer:
    def __init__(self, machine, lr=0.1):
        self.machine = machine
        self.lr = lr
    def fit(self):
        m = self.machine
        m.weights.assign_add(self.lr * m.grad_dict['weights'])
        m.visible_biases.assign_add(self.lr * m.grad_dict['visible_biases'])
        m.hidden_biases.assign_add(self.lr * m.grad_dict['hidden_biases'])

if __name__ == "__main__":
    for f in glob.glob("results/*.csv"):
        os.remove(f)

    optimizer = SimpleOptimizer(rbm, lr=0.1)
    rbm.train(data, optimizer)

    print("Max weight magnitude:", np.max(np.abs(rbm.weights.numpy())))
    print("Mean weight magnitude:", np.mean(np.abs(rbm.weights.numpy())))

    # 1 — show the weight matrix rows as receptive fields
    # each row of W is a hidden unit's learned feature detector
    plot_image_grid(
        rbm.weights.numpy(),   # shape (8, 16)
        image_shape=(4, 4),
        n_pictures=8,
        save=False             # set True to save as PDF
    )

    # 2 — sample from the trained model and show what it generates
    samples, probs, _ = rbm.parallel_sample(
        n_step_MC=100,
        n_chains=16,           # generate 16 samples
        p_0=0.5, p_1=0.5
    )
    plot_image_grid(
        np.array(samples),
        image_shape=(4, 4),
        n_pictures=16,
        save=False
    )

    # 3 — show input vs reconstruction side by side
    test_input = data['x_test'][0]   # grab one test pattern
    reconstruction, _, _ = rbm.parallel_sample(inpt=test_input.reshape(1, 16))
    plot_input_sample(
        test_input,
        reconstruction[0],
        image_shape=(4, 4),
        save=False
    )

    # 4 — reconstruction cross entropy with inline plot
    rbm.reconstruction_cross_entropy(
        data['x_test'][:5],
        plot=True
    )