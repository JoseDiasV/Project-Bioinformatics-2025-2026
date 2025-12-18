import torch
# print(torch.version.cuda) 
# to check if you have installed GPU-capable PyTorch,
# if it returns "None", then you only have CPU-only PyTorch
# uninstall it with: 
# pip uninstall torch torchvision torchaudio torchmetrics -y
# and then install GPU-capable PyTorch (for NVIDIA GPUs with CUDA 12.1 support) with:
# pip install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu121
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Local imports
from LSTM_pssm import LSTM
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

# ===== LSTM training =====
def train_LSTM(train_loader, n_epochs):
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
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # (B, window_size, num_classes)

            # Take central timestep to match labels
            center_idx = X_batch.shape[1] // 2
            outputs_center = outputs[:, center_idx, :]

            loss = criterion(outputs_center, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} average loss: {total_loss/len(train_loader):.4f}")
    return model

# ===== Feature extraction for Random Forest =====
def extract_features(model, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(data_loader, desc="Extracting features", leave=False):
            X_batch = X_batch.to(device)
            # take central timestep features
            feats = model(X_batch)[:, X_batch.shape[1] // 2, :]
            features_list.append(feats.cpu())
            labels_list.append(Y_batch)

    X_feats = torch.cat(features_list, dim=0).numpy()
    Y_labels = torch.cat(labels_list, dim=0).numpy()

    return X_feats, Y_labels


# ===== Evaluation =====
def evaluate_rf(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    return acc, prec, rec

# ===== Main =====
if __name__ == "__main__":
    n_classes = 3
    batch_size = 10
    window_size = 13
    n_epochs_lstm = 9
    rf_trees = 500

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

    # Train LSTM
    lstm_model = train_LSTM(train_loader, n_epochs=n_epochs_lstm)

    # Extract features from last LSTM layer
    X_train_feats, y_train_feats = extract_features(lstm_model, train_loader)
    X_test_feats, y_test_feats   = extract_features(lstm_model, test_loader)

    # Train Random Forest
    # n_jobs=-1 added for parallel computation
    rf_model = RandomForestClassifier(n_estimators=rf_trees, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_feats, y_train_feats)

    # RF predictions
    y_pred = rf_model.predict(X_test_feats)

    # Evaluate RF
    acc, prec, rec = evaluate_rf(y_test_feats, y_pred)

    print(f"LSTM-RF Test accuracy: {acc}")
    print(f"LSTM-RF Test precision: {prec}")
    print(f"LSTM-RF Test recall: {rec}")
