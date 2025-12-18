import torch
# print(torch.version.cuda) 
# to check if you have installed GPU-capable PyTorch,
# if it returns "None", then you only have CPU-only PyTorch
# uninstall it with: 
# pip uninstall torch torchvision torchaudio torchmetrics -y
# and then install GPU-capable PyTorch (for NVIDIA GPUs with CUDA 12.1 support) with:
# pip install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu121
# alternatively, if you're on conda you can install with:
# conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.1 -c pytorch -c nvidia
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Local imports
from LSTM_pssm import LSTM, Softmax
from SS_db import ss_db

if torch.cuda.is_available():
    # this improves LSTM speed for fixed input sizes (window_size is fixed)
    torch.backends.cudnn.benchmark = True

# ===== Data loading =====
def load_dataset(path, window_size, n_classes):
    dataset = ss_db()
    dataset.read_db(path)
    dataset.read_pssm_to_db()
    X, Y = dataset.to_torch_tensor_db(window_size=window_size, n_classes=n_classes)
    return X, Y

# ===== Base LSTM training =====
def train_LSTM(train_loader, n_classes, n_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTM(input_dim=20, num_classes=n_classes).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch+1}/{n_epochs}] of LSTM training")
        model.train()
        total_loss = 0
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            X_batch = X_batch.to(device)  # (B, window_size, 20)
            Y_batch = Y_batch.to(device)  # (B,)

            optimizer.zero_grad()
            outputs = model(X_batch)      # (B, window_size, num_classes)

            # ===== take central timestep to match labels =====
            center_idx = X_batch.shape[1] // 2
            outputs_center = outputs[:, center_idx, :]  # (B, num_classes)

            loss = criterion(outputs_center, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} average loss: {total_loss/len(train_loader):.4f}")
    return model

# ===== Evaluation =====
def evaluate_model(model_LSTM, test_loader, n_classes, softmax=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    prec = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes).to(device)
    rec = torchmetrics.Recall(task="multiclass", average='macro', num_classes=n_classes).to(device)

    model_LSTM.eval()
    if softmax:
        softmax.eval()

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(test_loader, desc="Evaluating", leave=False):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            outputs = model_LSTM(X_batch)  # (B, window_size, num_classes)

            # ===== Take central timestep =====
            center_idx = X_batch.shape[1] // 2
            outputs_center = outputs[:, center_idx, :]

            if softmax:
                outputs_center = softmax(outputs_center)

            _, preds = torch.max(outputs_center, dim=-1)
            acc.update(preds, Y_batch)
            prec.update(preds, Y_batch)
            rec.update(preds, Y_batch)

    return acc.compute(), prec.compute(), rec.compute()

# ===== Main =====
if __name__ == "__main__":
    n_classes = 3
    batch_size = 10
    window_size = 13
    n_epochs_lstm = 9

    # Load data
    data, labels = load_dataset(r"./astral_cull_DATA/cullatraldata.fa", window_size=window_size, n_classes=n_classes)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    # num_workers=4, pin_memory=True - for parallel CPU loading of batches, so that the GPU, if in use, doesn't wait for data
    # persistent_workers=True - allows DataLoader workers to remain alive between epochs to reduce worker startup overhead
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Train base LSTM
    model = train_LSTM(train_loader, n_classes=n_classes, n_epochs=n_epochs_lstm)

    # Evaluate base LSTM
    test_acc, test_prec, test_rec = evaluate_model(model, test_loader, n_classes=n_classes)
    print(f"Base LSTM Test accuracy: {test_acc}")
    print(f"Base LSTM Test precision: {test_prec}")
    print(f"Base LSTM Test recall: {test_rec}")
