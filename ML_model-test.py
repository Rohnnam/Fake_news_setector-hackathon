import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import os
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class DatasetAnalyzer:
    @staticmethod
    def analyze_dataset_sizes():
        """Analyze and print dataset sizes"""
        liar_sizes = {
            'train': 10269,
            'validation': 1284,
            'test': 1267
        }

        print("\nDataset Size Analysis")
        print("\n1. LIAR Dataset:")
        print(f"Training samples: {liar_sizes['train']}")
        print(f"Validation samples: {liar_sizes['validation']}")
        print(f"Test samples: {liar_sizes['test']}")
        print(f"Total LIAR samples: {sum(liar_sizes.values())}")

        return sum(liar_sizes.values())


class LIARDataset(Dataset):
    def __init__(self, statements: pd.Series, labels: List[int], tokenizer, max_length: int = 512):
        self.statements = statements.tolist()  # Store statements as a list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        statement = self.statements[idx]
        encoding = self.tokenizer(
            statement,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'][0],  # Remove batch dimension
            'attention_mask': encoding['attention_mask'][0],  # Remove batch dimension
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load LIAR dataset from a specified directory."""
    train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', names=[
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts',
        'false_counts', 'half_true_counts', 'mostly_true_counts',
        'pants_on_fire_counts', 'context'])
    val_df = pd.read_csv(os.path.join(data_dir, 'valid.tsv'), sep='\t', names=[
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts',
        'false_counts', 'half_true_counts', 'mostly_true_counts',
        'pants_on_fire_counts', 'context'])
    test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', names=[
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts',
        'false_counts', 'half_true_counts', 'mostly_true_counts',
        'pants_on_fire_counts', 'context'])

    label_map = {
        'true': 0, 'mostly-true': 0,
        'half-true': 1, 'barely-true': 1, 'false': 1, 'pants-fire': 1
    }

    for df in [train_df, val_df, test_df]:
        df['label'] = df['label'].map(label_map)

    return train_df, val_df, test_df


class FakeNewsDetector:
    def __init__(self, model_name: str = 'bert-base-uncased', dropout_prob: float = 0.2):  # Added dropout_prob
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob  # Use the same dropout for attention
        ).to(self.device)

    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced dataset"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights).to(self.device)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              batch_size: int = 32, epochs: int = 5, learning_rate: float = 2e-5,  # Increased batch size and epochs
              weight_decay: float = 0.01):
        """Train the model"""
        train_dataset = LIARDataset(
            train_df['statement'],
            train_df['label'].values,
            self.tokenizer
        )
        val_dataset = LIARDataset(
            val_df['statement'],
            val_df['label'].values,
            self.tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),  # Use all available CPU cores
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        class_weights = self.compute_class_weights(train_df['label'].values)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        scaler = GradScaler()

        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            all_train_labels = []
            all_train_preds = []

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)

                total_train_loss += loss.item()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Store training predictions and labels for calculating metrics
                train_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                train_labels = labels.cpu().numpy()
                all_train_labels.extend(train_labels)
                all_train_preds.extend(train_preds)

            self.model.eval()
            total_val_loss = 0
            all_val_labels = []
            all_val_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels) # Calculate validation loss
                    total_val_loss += loss.item()

                    val_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    val_labels = labels.cpu().numpy()
                    all_val_labels.extend(val_labels)
                    all_val_preds.extend(val_preds)

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            # Calculate metrics
            train_accuracy = accuracy_score(all_train_labels, all_train_preds)
            val_accuracy = accuracy_score(all_val_labels, all_val_preds)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                all_val_labels, all_val_preds, average='binary')

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Average training loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
                print("Saved new best model!")

    def predict(self, statements: List[str]) -> Tuple[List[int], List[np.ndarray]]:
        """Predict on new statements"""
        self.model.eval()
        statements_series = pd.Series(statements)
        dataset = LIARDataset(statements_series, [0] * len(statements), self.tokenizer)
        loader = DataLoader(dataset, batch_size=16)

        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return predictions, probabilities


def main():
    # Analyze dataset sizes
    dataset_analyzer = DatasetAnalyzer()
    dataset_analyzer.analyze_dataset_sizes()

    # Load data
    data_dir = r'C:\Users\Rohan Nambiar\Documents\Vscode\Project\data'  # Correct path
    train_df, val_df, test_df = load_data(data_dir=data_dir)

    print("\nFinal Dataset Sizes:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Initialize and train model
    detector = FakeNewsDetector()

    print("\nStarting model training...")
    detector.train(train_df, val_df, epochs=5, batch_size=32)  # Train with tuned hyperparameters

    # Load best model
    detector.model.load_state_dict(torch.load('best_model.pt'))
    detector.model.eval() # Set model to evaluation mode

    # Evaluate on the test set
    test_dataset = LIARDataset(test_df['statement'], test_df['label'].values, detector.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=os.cpu_count(), pin_memory=True)

    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(detector.device)
            attention_mask = batch['attention_mask'].to(detector.device)
            labels = batch['labels'].to(detector.device)

            outputs = detector.model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_labels = labels.cpu().numpy()

            all_test_labels.extend(test_labels)
            all_test_preds.extend(test_preds)

    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_test_labels, all_test_preds, average='binary')

    print("\nTest Set Evaluation:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    # Test prediction (sample statements)
    sample_statements = [
        "Scientists confirm new vaccine is safe and effective",
        "SHOCKING: Government conspiracy revealed in leaked documents!!!"
    ]

    predictions, probabilities = detector.predict(sample_statements)

    print("\nSample Predictions:")
    for statement, pred, prob in zip(sample_statements, predictions, probabilities):
        print(f"\nStatement: {statement}")
        print(f"Prediction: {'Fake' if pred == 1 else 'True'}")
        print(f"Confidence: {max(prob):.4f}")


if __name__ == "__main__":
    main()
