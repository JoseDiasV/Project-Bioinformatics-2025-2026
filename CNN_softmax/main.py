import torch
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

from tqdm import tqdm

from sklearn.model_selection import train_test_split

# local classes
from CNN_pssm import CNN, Softmax
from SS_db import ss_db

if __name__ == "__main__":

    window_size = 13
    n_classes = 3
    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"


    dataset = ss_db()
    dataset.read_db('test_db.fa')
    #dataset.read_db('top100lines.fa')
    dataset.read_pssm_to_db()
    data, labels = dataset.to_torch_tensor_db(window_size=13)


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

    model = CNN(in_channels=1, num_classes=n_classes, window_size=window_size).to(device)
    print(model)

    ###############################################
    ################# CNN training ################
    ###############################################

    ## applies SoftMax internally, no need 
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=10 # paper says 10
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

    

    # testing
    acc = torchmetrics.Accuracy(task="multiclass",num_classes=n_classes)
    prec = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes)
    rec = torchmetrics.Recall(task="multiclass", average='macro', num_classes=n_classes)

    # Iterate over the dataset batches
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            # Get predicted probabilities for test data batch
            #outputs = nn.Softmax(model(data), dim=1)

            outputs = model(data) # works
            #print(outputs)
            _, preds = torch.max(outputs, 1)  # preds are indeces of max values == classes
            #print(_, preds)
            acc.update(preds, labels)
            prec.update(preds, labels)
            rec.update(preds, labels)

    #Compute total test accuracy
    test_accuracy = acc.compute()
    test_precisionn = prec.compute()
    test_recall = rec.compute()

    print(f"Test accuracy CNN: {test_accuracy}")
    print(f"Test precision CNN: {test_precisionn}")
    print(f"Test recall CNN: {test_recall}")

    #################### ReLU extraction ######################
    #### training data for Softmax classifier##################

    features = []
    labels = []
    with torch.no_grad():
        for data, label in train_loader:

            data = data.to(device)
            ReLU = model(data, ReLU_out=True)
            features.append(ReLU)
            labels.append(label)

    features_torch = torch.cat(features, dim = 0)
   # print(features_torch.shape[1])
    labels_torch = torch.cat(labels, dim = 0)

    ReLU_train_dataset = TensorDataset(features_torch, labels_torch)

    ###############################################
    ######## Softmax classifier training ##########
    ###############################################
    
    ReLU_train_loader = DataLoader(dataset=ReLU_train_dataset, batch_size=batch_size)

    feature_count = ReLU_train_dataset.tensors[0].shape[1] # the length of one ReLU linearized vector
    model_softmax = Softmax(feature_count, n_classes)
    model_softmax.to(device)
    model_softmax.state_dict()


     # define loss, optimizier
    optimizer = torch.optim.SGD(model_softmax.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    # Train the model
    Loss = []
    n_epochs = 100
    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + 1}/{n_epochs}] of Softmax classifier training")
        for features, targets in tqdm(ReLU_train_loader):

            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            scores = model_softmax(features)
            loss = criterion(scores, targets)
            Loss.append(loss.item())
            loss.backward()
            optimizer.step()


    ############################################
    ############## CNN-S evaluation ############
    ############################################

    acc_CNNs = torchmetrics.Accuracy(task="multiclass",num_classes=n_classes)
    prec_CNNS = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes)
    rec_CNNs = torchmetrics.Recall(task="multiclass", average='macro', num_classes=n_classes)


    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            # Get predicted probabilities for test data batch
            ReLU_output = model(data, ReLU_out = True) # works
            #print(outputs)
            outputs = model_softmax(ReLU_output)
            _, preds = torch.max(outputs, 1)  # preds are indeces of max values == classes
            #print(_, preds)
            acc_CNNs.update(preds, labels)
            prec_CNNS.update(preds, labels)
            rec_CNNs.update(preds, labels)

    #Compute total test accuracy
    test_accuracy_CNNs = acc_CNNs.compute()
    test_precision_CNNs = prec_CNNS.compute()
    test_recall_CNNs = rec_CNNs.compute()

    print(f"Test accuracy CNNs: {test_accuracy_CNNs}")
    print(f"Test precision CNNs: {test_precision_CNNs}")
    print(f"Test recall CNNS: {test_recall_CNNs}")

    print(f"\nTest accuracy CNN: {test_accuracy}")
    print(f"Test precision CNN: {test_precisionn}")
    print(f"Test recall CNN: {test_recall}")
