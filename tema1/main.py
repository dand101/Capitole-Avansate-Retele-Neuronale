from dataset import load_data
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def init_weights(input, hidden, output):
    w1 = torch.randn(input, hidden, device=device) * torch.sqrt(torch.tensor(2.0 / input, device=device))
    b1 = torch.zeros(hidden, device=device)

    w2 = torch.randn(hidden, output, device=device) * torch.sqrt(torch.tensor(2.0 / hidden, device=device))
    b2 = torch.zeros(output, device=device)

    return w1, b1, w2, b2


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def relu(z):
    return torch.maximum(z, torch.tensor(0.0, device=device))


def relu_derivative(z):
    return (z > 0).float()


def forward_pass(inp, w1, b1, w2, b2):
    z1 = inp @ w1 + b1
    a1 = relu(z1)

    z2 = a1 @ w2 + b2

    return z1, a1, z2


def backpropagation(inp, target, z1, a1, z2, w1, w2, b1, b2, lr=0.01):
    batch_size = target.shape[0]

    # if batch_size < 64:
    #     print(batch_size)

    one_hot = torch.zeros(batch_size, 10, device=device)

    one_hot[torch.arange(len(target)), target] = 1

    softmax_output = torch.exp(z2) / torch.sum(torch.exp(z2), dim=1, keepdim=True)
    dz2 = softmax_output - one_hot

    dw2 = a1.T @ dz2
    db2 = torch.sum(dz2, dim=0)

    da1 = dz2 @ w2.T
    dz1 = da1 * relu_derivative(a1)

    dw1 = inp.T @ dz1
    db1 = torch.sum(dz1, dim=0)

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    return w1, b1, w2, b2


def validation_accuracy_and_loss(val_x, val_y, w1, w2, b1, b2):
    z1, a1, z2 = forward_pass(val_x, w1, b1, w2, b2)

    predictions = torch.argmax(z2, dim=1)
    correct = (predictions == val_y).sum().item()

    loss = torch.nn.CrossEntropyLoss()(z2, val_y)

    accuracy = correct / len(val_x) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Validation Loss: {loss.item():.4f}")


def train(w1, w2, b1, b2, train_x, train_y, val_x, val_y, epochs=5, lr=0.01, batch_size=64):
    nr_images = len(train_x)

    for epoch in range(epochs):
        random_permutation = torch.randperm(nr_images)
        total_loss = 0
        correct = 0

        for i in range(0, nr_images, batch_size):
            ind = random_permutation[i:i + batch_size]
            batch_x, batch_y = train_x[ind], train_y[ind]

            z1, a1, z2 = forward_pass(batch_x, w1, b1, w2, b2)

            loss = torch.nn.CrossEntropyLoss()(z2, batch_y)
            total_loss += loss.item()

            predictions = torch.argmax(z2, dim=1)
            correct += (predictions == batch_y).sum().item()

            w1, b1, w2, b2 = backpropagation(batch_x, batch_y, z1, a1, z2, w1, w2, b1, b2, lr=lr)

        print(
            f"\nEpoch {epoch + 1}/{epochs} \nLoss: {total_loss / (nr_images / batch_size):.4f} \nAccuracy: {correct / nr_images * 100:.2f}%")
        validation_accuracy_and_loss(val_x, val_y, w1, w2, b1, b2)
    return w1, b1, w2, b2


def evaluation(w1, w2, b1, b2, test_x, test_y, batch_size=64):
    nr_images = len(test_x)
    correct = 0
    total_loss = 0

    for i in range(0, nr_images, batch_size):
        batch_x, batch_y = test_x[i:i + batch_size], test_y[i:i + batch_size]

        z1, a1, z2 = forward_pass(batch_x, w1, b1, w2, b2)

        loss = torch.nn.functional.cross_entropy(z2, batch_y)
        total_loss += loss.item()

        predictions = torch.argmax(z2, dim=1)
        correct += (predictions == batch_y).sum().item()

    accuracy = correct / nr_images * 100
    print("\n" * 2)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {total_loss / (nr_images / batch_size):.4f}")


(train_x, train_y), (validation_x, validation_y), (test_x, test_y) = load_data()

w1, b1, w2, b2 = init_weights(28 * 28, 100, 10)
w1, b1, w2, b2 = train(w1, w2, b1, b2, train_x, train_y, validation_x, validation_y, epochs=5, lr=0.01, batch_size=63)
evaluation(w1, w2, b1, b2, test_x, test_y, batch_size=64)
