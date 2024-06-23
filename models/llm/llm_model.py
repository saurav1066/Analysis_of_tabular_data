import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def encode_data(df, tokenizer, max_len=512):
    """
    Converts tabular data into a format suitable for BERT, which involves tokenizing the text.
    """
    inputs = []
    labels = []
    for _, row in df.iterrows():
        # Convert the row to text by concatenating values with column names as keys
        text = ' '.join([f"{key}: {value}" for key, value in row.items() if key != 'target'])
        label = row['target']
        inputs.append(tokenizer(text, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt"))
        labels.append(label)

    # Create tensors for the inputs and labels
    input_ids = torch.cat([x['input_ids'] for x in inputs], dim=0)
    attention_mask = torch.cat([x['attention_mask'] for x in inputs], dim=0)
    labels = torch.tensor(labels)

    return TensorDataset(input_ids, attention_mask, labels)


def create_data_loaders(dataset, batch_size=16):
    """
    Create PyTorch DataLoaders for training and validation.
    """
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, learning_rate=5e-5, epochs=3):
    """
    Trains the BERT model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Loss {loss.item()}")

        # Evaluate the model
        model.eval()
        val_accuracy = []
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs[1]
            predictions = torch.argmax(logits, dim=-1)
            accuracy = accuracy_score(batch[2].cpu(), predictions.cpu())
            val_accuracy.append(accuracy)
        print(f"Validation Accuracy: {sum(val_accuracy) / len(val_accuracy):.2f}")


# Example usage
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/heart_disease/heart_disease_uci.csv")  # Ensure the dataset is loaded correctly
    dataset = encode_data(df, tokenizer)
    train_loader, val_loader = create_data_loaders(dataset)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=2)  # Adjust num_labels as necessary
    train_model(model, train_loader, val_loader)
