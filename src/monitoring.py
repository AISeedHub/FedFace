"""Monitor FL Training Progress"""

import json
import matplotlib.pyplot as plt


def plot_training_history(log_file="fl_history.json"):
    """Plot federated learning training history."""
    with open(log_file, 'r') as f:
        history = json.load(f)

    rounds = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rounds, history['accuracy'], 'b-o')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning Accuracy')
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(rounds, history['loss'], 'r-o')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Federated Learning Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('fl_training_progress.png')
    plt.show()


if __name__ == "__main__":
    plot_training_history()