from dataset import load_data
import torch
import torch.nn.functional as functional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def init_weights(input, hidden, output):
    w1 = torch.randn(input, hidden, device=device) * torch.sqrt(torch.tensor(2.0 / input, device=device))
    b1 = torch.zeros(hidden, device=device)

    w2 = torch.randn(hidden, output, device=device) * torch.sqrt(torch.tensor(2.0 / hidden, device=device))
    b2 = torch.zeros(output, device=device)

    return w1, b1, w2, b2


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return torch.maximum(x, torch.tensor(0.0, device=device))


def relu_derivative(x):
    return (x > 0).float()


def forward_pass(inp, w1, b1, w2, b2):
    p_activ_hid = inp @ w1 + b1
    activ_hid = relu(p_activ_hid)

    p_activ_out = activ_hid @ w2 + b2

    return p_activ_hid, activ_hid, p_activ_out


def backpropagation(inp, target, p_activ_hid, activ_hid, p_activ_out, w1, w2, b1, b2, lr=0.01):
    batch_size = target.shape[0]

    # if batch_size < 64:
    #     print(batch_size)

    one_hot = torch.zeros(batch_size, 10, device=device)
    one_hot[torch.arange(len(target)), target] = 1

    softmax_output = torch.exp(p_activ_out) / torch.sum(torch.exp(p_activ_out), dim=1, keepdim=True)
    err_output = softmax_output - one_hot

    grad_w2 = activ_hid.T @ err_output
    grad_b2 = torch.sum(err_output, dim=0)

    err_hidden = err_output @ w2.T
    err_hidden *= relu_derivative(p_activ_hid)

    grad_w1 = inp.T @ err_hidden
    grad_b1 = torch.sum(err_hidden, dim=0)

    w1 -= lr * grad_w1
    b1 -= lr * grad_b1
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2

    return w1, b1, w2, b2


def validation_accuracy_and_loss(val_x, val_y, w1, w2, b1, b2):
    p_activ_hid, activ_hid, p_activ_out = forward_pass(val_x, w1, b1, w2, b2)

    predictions = torch.argmax(p_activ_out, dim=1)
    correct = (predictions == val_y).sum().item()

    loss = functional.cross_entropy(p_activ_out, val_y)

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

            p_activ_hid, activ_hid, p_activ_out = forward_pass(batch_x, w1, b1, w2, b2)

            loss = functional.cross_entropy(p_activ_out, batch_y)
            total_loss += loss.item()

            predictions = torch.argmax(p_activ_out, dim=1)
            correct += (predictions == batch_y).sum().item()

            w1, b1, w2, b2 = backpropagation(batch_x, batch_y, p_activ_hid, activ_hid, p_activ_out, w1, w2, b1, b2, lr=lr)

        print(
            f"\nEpoch {epoch + 1}/{epochs} \nLoss: {total_loss / nr_images:.4f} \nAccuracy: {correct / nr_images * 100:.2f}%")
        validation_accuracy_and_loss(val_x, val_y, w1, w2, b1, b2)
    return w1, b1, w2, b2


def evaluation(w1, w2, b1, b2, test_x, test_y, batch_size=64):
    nr_images = len(test_x)
    correct = 0
    total_loss = 0

    for i in range(0, nr_images, batch_size):
        batch_x, batch_y = test_x[i:i + batch_size], test_y[i:i + batch_size]

        p_activ_hid, activ_hid, p_activ_out = forward_pass(batch_x, w1, b1, w2, b2)

        loss = torch.nn.functional.cross_entropy(p_activ_out, batch_y)
        total_loss += loss.item()

        predictions = torch.argmax(p_activ_out, dim=1)
        correct += (predictions == batch_y).sum().item()

    accuracy = correct / nr_images * 100
    print("\n" * 2)
    print("Final Results:")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {total_loss / nr_images:.4f}")


(train_x, train_y), (validation_x, validation_y), (test_x, test_y) = load_data()

w1, b1, w2, b2 = init_weights(28 * 28, 100, 10)
w1, b1, w2, b2 = train(w1, w2, b1, b2, train_x, train_y, validation_x, validation_y, epochs=10, lr=0.01, batch_size=64)
evaluation(w1, w2, b1, b2, test_x, test_y, batch_size=64)
