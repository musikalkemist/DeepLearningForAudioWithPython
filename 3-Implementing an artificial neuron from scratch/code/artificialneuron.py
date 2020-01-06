import math

def sigmoid(x):
    """Sigmoid activation function
    Args:
        x (float): Value to be processed
    Returns:
        y (float): Output
    """
    y = 1.0 / (1 + math.exp(-x))
    return y


def activate(inputs, weights):
    """Computes activation of neuron based on input signals and connection
    weights. Output = f(x_1*w_1 + x_2*w_2 + ... + x_k*w_k), where 'f' is the
    sigmoid function.
    Args:
        inputs (list): Input signals
        weights (list): Connection weights
    Returns:
        output (float): Output value
    """

    h = 0

    # compute the sum of the product of the input signals and the weights
    # here we're using pythons "zip" function to iterate two lists together
    for x, w in zip(inputs, weights):
        h += x*w

    # process sum through sigmoid activation function
    return sigmoid(h)


if __name__ == "__main__":
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4, 0.7, 0.2]
    output = activate(inputs, weights)
    print(output)