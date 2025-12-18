import torch
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold

# local classes
from CNN_pssm import CNN, Softmax
from SS_db import ss_db

import time

def load_dataset(path, w_size):

    dataset = ss_db()
    dataset.read_db(path)

    dataset.read_pssm_to_db_exc()
    data, labels = dataset.to_torch_tensor_db(window_size=w_size)
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

def evaluate_model(model_CNN, test_loader, softmax = None):

    acc = torchmetrics.Accuracy(task="multiclass",num_classes=n_classes)
    prec = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes)
    rec = torchmetrics.Recall(task="multiclass", average='macro', num_classes=n_classes)

    # Iterate over the dataset batches
    model_CNN.eval()
    if softmax:
        softmax.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            # Get predicted probabilities for test data batch
            #outputs = nn.Softmax(model(data), dim=1)
            if softmax:
                ReLU_output = model_CNN(data, ReLU_out = True) # works
                outputs = softmax(ReLU_output)
            else:
                outputs = model_CNN(data) # works
            
            _, preds = torch.max(outputs, 1)  # preds are indeces of max values == classes
            #print(_, preds)
            acc.update(preds, labels)
            prec.update(preds, labels)
            rec.update(preds, labels)

    #Compute total test accuracy
    test_accuracy = acc.compute()
    test_precisionn = prec.compute()
    test_recall = rec.compute()

    return test_accuracy, test_precisionn, test_recall

def ReLU_extractor(model, data_loader):
    features = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:

            data = data.to(device)
            ReLU = model(data, ReLU_out=True)
            features.append(ReLU)
            labels.append(label)

    features_torch = torch.cat(features, dim = 0)
   # print(features_torch.shape[1])
    labels_torch = torch.cat(labels, dim = 0)

    ReLU_train_dataset = TensorDataset(features_torch, labels_torch)

    return ReLU_train_dataset

def train_Softmax(data_loader, feature_count, n_classes, n_epochs=100):
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

if __name__ == "__main__":

    start_time = time.perf_counter()


    window_size = 13
    n_classes = 3
    batch_size = 20
    n_fold_CV = 10
    n_epochs_CNN = 10
    n_epochs_softmax = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


    ################ Data loading ###############

    data, labels = load_dataset('cullatraldata.txt', window_size)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # ⭐ keeps class balance
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    ## model is CNN trained on input dataloader  
    model = train_CNN(train_loader=train_loader, n_classes=n_classes, w_size=window_size, n_epochs=n_epochs_CNN)
    
    # skf = StratifiedKFold( # keeps class balance
    #         n_splits=n_fold_CV,
    #         shuffle=True,
    #         random_state=42
    #     )
    

    # accs_CNNs = []
    # # repeat for cross validation
    # for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):

    #     print(f"\n===== Fold {fold + 1}/{n_fold_CV} =====")
    
    #     X_train = data[train_idx]
    #     y_train = labels[train_idx]

    #     X_test  = data[test_idx]
    #     y_test  = labels[test_idx]

    #     train_dataset = TensorDataset(X_train, y_train)
    #     test_dataset  = TensorDataset(X_test, y_test)

    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    #     ## model is CNN trained on input dataloader  
    #     model = train_CNN(train_loader=train_loader, n_classes=n_classes, w_size=window_size, n_epochs=n_epochs_CNN)

    #     ############################################
    #     ############### CNN evaluation #############
    #     ############################################

    #     ## evaluation of CNN
    #     test_accuracy, test_precisionn, test_recall = evaluate_model(model_CNN=model, test_loader=test_loader)

    #     #################### ReLU extraction ######################
    #     ######## training data for Softmax classifier #############
    #     ###########################################################

    #     ReLU_train_dataset = ReLU_extractor(model=model, data_loader=train_loader)

    #     ###############################################
    #     ######## Softmax classifier training ##########
    #     ###############################################
        
    #     ReLU_train_loader = DataLoader(dataset=ReLU_train_dataset, batch_size=batch_size)

    #     feature_count = ReLU_train_dataset.tensors[0].shape[1] # the length of one ReLU linearized vector

    #     model_softmax = train_Softmax(ReLU_train_loader, feature_count, n_classes, n_epochs=n_epochs_softmax)

    #     ############################################
    #     ############## CNN-S evaluation ############
    #     ############################################

    #     test_accuracy_CNNs, test_precision_CNNs, test_recall_CNNs = evaluate_model(model_CNN=model, test_loader=test_loader, softmax=model_softmax)

    #     print(f"Test accuracy CNNs: {test_accuracy_CNNs}")
    #     print(f"Test precision CNNs: {test_precision_CNNs}")
    #     print(f"Test recall CNNS: {test_recall_CNNs}")

    #     print(f"\nTest accuracy CNN: {test_accuracy}")
    #     print(f"Test precision CNN: {test_precisionn}")
    #     print(f"Test recall CNN: {test_recall}")

    #     accs_CNNs.append(test_accuracy_CNNs)

    # print(accs_CNNs)

    # end_time = time.perf_counter()

    # print(f"Time needed for:\n\t{n_fold_CV}-fold crossvalidation\n\t{n_epochs_CNN} epochs of CNN training\n\t{n_epochs_softmax} epochs of SoftMax training\n\n\t is {end_time - start_time} seconds")