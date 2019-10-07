import numpy as np


def pad_axis0(array, value):
    return np.pad(array, pad_width=(0,1), mode='constant', constant_values=value)


def shift(array):
        return pad_axis0(array, 0)[1:]


def calculate_lambda_returns(rewards, qvalues, dones, mask, discount, lambd):
    dones = dones.astype(np.float32)
    qvalues[-1] *= (1.0 - dones[-1])
    lambda_returns = rewards + (discount * qvalues[1:])
    for i in reversed(range(len(rewards) - 1)):
        a = lambda_returns[i] + (discount * lambd * mask[i]) * (lambda_returns[i+1] - qvalues[i+1])
        b = rewards[i]
        lambda_returns[i] = (1.0 - dones[i]) * a + dones[i] * b
    return lambda_returns


def calculate_nstep_returns(rewards, qvalues, dones, discount, n):
    # Counterintuitively, the bootstrap is treated is as a reward too
    rewards = pad_axis0(rewards, qvalues[-1])
    dones   = pad_axis0(dones, 1.0)

    mask    = np.ones_like(rewards)
    decay   = 1.0
    returns = np.copy(rewards)

    for i in range(n):
        decay *= discount
        mask *= (1.0 - dones)

        rewards = shift(rewards)
        qvalues = shift(qvalues)
        dones   = shift(dones)

        if i != (n-1):
            returns += (mask * decay * rewards)
        else:
            returns += (mask * decay * qvalues)

    return returns[:-1]  # Remove bootstrap placeholder
