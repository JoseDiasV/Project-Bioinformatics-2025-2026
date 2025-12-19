import torch
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset
import torchmetrics


from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

# local classes
from models_pssm import CNN, Softmax, LSTM
from SS_db import ss_db

import numpy as np

def load_dataset(path, w_size, n_classes=3):

    dataset = ss_db()
    dataset.read_db(path)

    dataset.read_pssm_to_db()
    data, labels = dataset.to_torch_tensor_db(window_size=w_size, n_classes=n_classes)
    return data, labels

def train_CNN(train_loader, n_classes, w_size, n_epochs = 10):
    # define the CNN
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN(in_channels=1, num_classes=n_classes, window_size=w_size).to(device)
    print(model)

    ###############################################
    ################# CNN training ################
    ###############################################

    ## applies SoftMax internally, no need 
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=n_epochs # paper says 10 -> default 10
    for epoch in range(num_epochs):
    # Iterate over training batches
        print(f"Epoch [{epoch + 1}/{num_epochs}] of CNN training")

        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            #print(targets)
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            #print(scores)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model_CNN, test_loader, n_classes=3, softmax = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    acc = torchmetrics.Accuracy(task="multiclass",num_classes=n_classes).to(device)
    prec = torchmetrics.Precision(task="multiclass", average=None, num_classes=n_classes).to(device)
    rec = torchmetrics.Recall(task="multiclass", average='macro', num_classes=n_classes).to(device)
    class_pred_probs = []

    # Iterate over the dataset batches
    model_CNN.eval()
    if softmax:
        softmax.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            # Get predicted probabilities for test data batch
            #outputs = nn.Softmax(model(data), dim=1)
            if softmax:
                ReLU_output = model_CNN(data, ReLU_out = True) # works
                outputs = softmax(ReLU_output)
                # print(outputs)
            else:
                outputs = model_CNN(data) # works

            outputs_probs = torch.softmax(outputs, dim=1)
            #print(outputs_probs)

            class_pred_probs.append(outputs_probs)
            _, preds = torch.max(outputs, 1)  # preds are indeces of max values == classes
            preds = preds.to(device)
            
            acc.update(preds, labels)
            prec.update(preds, labels)
            rec.update(preds, labels)

    #Compute total test accuracy
    test_accuracy = acc.compute()
    test_precisionn = prec.compute()
    test_recall = rec.compute()

    # creates a matrix-like of (n_samples,n_classes)
    class_pred_probs = torch.cat(class_pred_probs, dim=0)

    return class_pred_probs, test_accuracy, test_precisionn, test_recall

def ReLU_extractor(model, data_loader):
    features = []
    labels = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            ReLU = model(data, ReLU_out=True)
            features.append(ReLU)
            labels.append(label)

    features_torch = torch.cat(features, dim = 0)
   # print(features_torch.shape[1])
    labels_torch = torch.cat(labels, dim = 0)

    ReLU_train_dataset = TensorDataset(features_torch, labels_torch)

    return ReLU_train_dataset

def train_Softmax(data_loader, feature_count, n_classes, n_epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_softmax = Softmax(feature_count, n_classes)
    model_softmax.to(device)
    model_softmax.state_dict()


     # define loss, optimizier
    optimizer = torch.optim.SGD(model_softmax.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    # Train the model
    Loss = []

    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1}/{n_epochs}] of Softmax classifier training")
        for features, targets in tqdm(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            scores = model_softmax(features)
            loss = criterion(scores, targets)
            Loss.append(loss.item())
            loss.backward()
            optimizer.step()

    return model_softmax

def train_LSTM(train_loader, n_epochs, n_classes=3):

    if torch.cuda.is_available():
    # this improves LSTM speed for fixed input sizes (window_size is fixed)
        torch.backends.cudnn.benchmark = True

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

def evaluate_rf(y_true, y_pred, n_classes=3):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, labels=range(n_classes))
    rec = recall_score(y_true, y_pred, average='macro')
    return acc, prec, rec

def ensemble_prediction(pred_CNNs, pred_LSTMrf, weight_CNNs = 0.5, weight_LSTM_rf = 0.5):

    ## same formatting ##
    pred_CNNs = pred_CNNs.cpu().numpy()
    pred_LSTMrf = np.asarray(pred_LSTMrf)

    # print(pred_CNNs.shape, pred_LSTMrf.shape)

    assert pred_CNNs.shape == pred_LSTMrf.shape, "Shape mismatch"

    combined_scores = ( weight_CNNs * pred_CNNs + weight_LSTM_rf * pred_LSTMrf )

    return combined_scores