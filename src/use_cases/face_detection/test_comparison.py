import os
import sys
import matplotlib.pyplot as plt
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.use_cases.face_detection.models.cnn import SimpleCNN
from src.use_cases.face_detection.models.mlp import MLP
from src.use_cases.face_detection.models.resnet import PretrainedResNet
from src.use_cases.face_detection.utils.lfw_100_loader import (
    LFW100Dataset,
    LFW100EmbDataset,
)

MODELS = {
    "mlp": MLP,
    "resnet": PretrainedResNet,
    "cnn": SimpleCNN,
}

LOADER = {
    "folder": LFW100Dataset,
    "npz": LFW100EmbDataset,  # Placeholder for potential future dataset loaders
}


class FaceClassification:
    """Face Classification"""

    def __init__(self, config: dict):

        # Initialize model

        self.model = MODELS[config["model"]["name"]](
            num_classes=config["model"]["num_classes"]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.config = config

        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["learning_rate"], momentum=0.9
        )

        # Load client data
        self.train_loader, self.test_loader = self._load_data()

    def _load_data(self):
        """Load client-specific data"""
        data_path = os.path.join(self.config["test_data_path"])
        print("Loading data from: ", data_path, "\n\n")

        dataset = LOADER[self.config["data_type"]](data_dir=data_path, split="train")
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["batch_size"], shuffle=False
        )

        return train_loader, test_loader

    def train_model(self, epochs: int) -> dict[str, float]:
        """Train model locally"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        print(f"Initialized with {len(self.train_loader.dataset)} training samples")
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            total_loss += epoch_loss
            print(
                f" Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}"
            )

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / (epochs * len(self.train_loader))

        print(f" Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    def evaluate_model(self) -> tuple[float, float, dict]:
        """Evaluate model"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        print(f"Evaluate with {len(self.test_loader.dataset)} testing samples")
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = test_loss / len(self.test_loader)

        print(f" Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy, {"test_accuracy": accuracy}


def generate_comparison_report(central_result, federated_result, config):
    """Generate detailed comparison report"""

    # Extract results
    central_loss, central_acc, central_metrics = central_result
    fed_loss, fed_acc, fed_metrics = federated_result

    # Calculate differences
    acc_difference = central_acc - fed_acc
    loss_difference = central_loss - fed_loss
    acc_improvement_percent = (acc_difference / fed_acc) * 100 if fed_acc != 0 else 0

    # Generate text report
    report = f"""
📊 MODEL PERFORMANCE COMPARISON REPORT
{'=' * 60}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of Classes: {config['model']['num_classes']}
"""


    return report, {
        'central_accuracy': central_acc,
        'federated_accuracy': fed_acc,
        'central_loss': central_loss,
        'federated_loss': fed_loss,
        'accuracy_difference': acc_difference,
        'loss_difference': loss_difference,
        'accuracy_improvement_percent': acc_improvement_percent,
        'better_model': 'centralized' if acc_difference > 0 else 'federated' if acc_difference < 0 else 'equal',
        'performance_grade': 'excellent' if min(central_acc, fed_acc) > 90 else 'good' if min(central_acc,
                                                                                              fed_acc) > 80 else 'needs_improvement',
        'timestamp': datetime.now().isoformat(),
        'model_config': config['model']
    }


def create_comparison_chart(metrics_data):
    """Create visual comparison charts"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    models = ['Centralized', 'Federated']
    accuracies = [metrics_data['central_accuracy'], metrics_data['federated_accuracy']]
    losses = [metrics_data['central_loss'], metrics_data['federated_loss']]
    colors = ['#2E86AB', '#A23B72']

    # Accuracy comparison bar chart
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Loss comparison bar chart
    bars2 = ax2.bar(models, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.05,
                 f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save chart
    chart_filename = f"comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Comparison chart saved: {chart_filename}")

    return chart_filename


def create_performance_summary_table(metrics_data):
    """Create a summary table of key metrics"""

    summary = f"""
📋 PERFORMANCE SUMMARY TABLE
{'=' * 50}
{'Metric':<25} {'Centralized':<15} {'Federated':<15} {'Difference':<15}
{'-' * 70}
{'Accuracy (%)':<25} {metrics_data['central_accuracy']:<15.2f} {metrics_data['federated_accuracy']:<15.2f} {metrics_data['accuracy_difference']:<+15.2f}
{'Loss':<25} {metrics_data['central_loss']:<15.4f} {metrics_data['federated_loss']:<15.4f} {metrics_data['loss_difference']:<+15.4f}
{'Better Model':<25} {metrics_data['better_model']:<30}
{'-' * 70}
"""
    return summary


def save_report_to_file(report_text, metrics_data, summary_table):
    """Save comprehensive report to files"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save complete text report
    txt_filename = f"model_comparison_report_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
        f.write("\n\n")
        f.write(summary_table)

    # Save detailed JSON data
    json_filename = f"comparison_metrics_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    # Save CSV for easy analysis
    csv_filename = f"comparison_data_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("Model,Accuracy,Loss\n")
        f.write(f"Centralized,{metrics_data['central_accuracy']},{metrics_data['central_loss']}\n")
        f.write(f"Federated,{metrics_data['federated_accuracy']},{metrics_data['federated_loss']}\n")

    return txt_filename, json_filename, csv_filename


def load_config(config_path="src/use_cases/face_detection/configs/base.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def main():
    """Start federated learning client"""

    print("🌸 FedFlower - Face Classification Comparison")
    print("=" * 50)

    # Load configuration
    config = load_config()

    print("=" * 50)
    print("Centralize Model Test: ")
    model_path = "centralize_model.pth"
    central_run = FaceClassification(config)
    # load model weight to test
    central_run.model.load_state_dict(torch.load(model_path))
    central_result = central_run.evaluate_model()  # avg_loss, accuracy, {"test_accuracy": accuracy}
    print("=" * 50)

    print("Federated Model Test: ")
    fed_model_path = "client_0_model.pth"
    federated_run = FaceClassification(config)
    # load model weight to test
    federated_run.model.load_state_dict(torch.load(fed_model_path))
    federated_result = federated_run.evaluate_model()
    print("=" * 50)

    print("GENERATING COMPREHENSIVE COMPARISON REPORT...")
    print("=" * 50)

    # Generate detailed comparison report
    report_text, metrics_data = generate_comparison_report(central_result, federated_result, config)

    # Create performance summary table
    summary_table = create_performance_summary_table(metrics_data)

    # Display report in console
    print(report_text)
    print(summary_table)

    # Create visual comparison charts
    create_comparison_chart(metrics_data)

    print(f"\n🎯 QUICK SUMMARY:")
    print(f"   • Better Model: {metrics_data['better_model'].capitalize()}")
    print(f"   • Accuracy Gap: {abs(metrics_data['accuracy_difference']):.2f}%")

if __name__ == "__main__":
    main()