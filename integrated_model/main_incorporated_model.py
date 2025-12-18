from header import load_dataset_CNN, train_CNN, train_Softmax, ReLU_extractor, evaluate_model, train_LSTM, extract_features, evaluate_rf
# from main_LSTM_RF import train_LSTM, extract_features, evaluate_rf

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import time


if __name__ == "__main__":

    start_time = time.perf_counter()

    window_size = 13
    n_classes = 3
    batch_size = 20
    n_fold_CV = 3
    n_epochs_CNN = 2
    n_epochs_softmax = 100
    n_epochs_lstm = 2
    rf_trees = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"


    ################ Data loading ###############

    #data, labels = load_dataset('test_db.fa', window_size)
    data, labels = load_dataset_CNN('cullatraldata_one_third.txt', window_size)
    #data, labels = load_dataset_CNN('test_db.fa', window_size, n_classes=n_classes)

    # input format for CNN is (batchsize, 1, W, 20) for LSTM (batchsize, W, 20)
    data_CNN = data.unsqueeze(1)

    skf = StratifiedKFold( # keeps class balance
            n_splits=n_fold_CV,
            shuffle=True,
            random_state=42
        )
    

    accs_CNNs = []
    # repeat for cross validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):

        print(f"\n===== Fold {fold + 1}/{n_fold_CV} =====")


        X_CNN_train = data_CNN[train_idx]
        X_LSTM_train = data[train_idx]
        y_train = labels[train_idx]

        X_CNN_test  = data_CNN[test_idx]
        X_LSTM_test = data[test_idx]
        y_test  = labels[test_idx]

        train_dataset_CNN = TensorDataset(X_CNN_train, y_train)
        test_dataset_CNN  = TensorDataset(X_CNN_test, y_test)

        train_dataset_LSTM = TensorDataset(X_LSTM_train, y_train)
        test_dataset_LSTM  = TensorDataset(X_LSTM_test, y_test)

        train_loader_CNN = DataLoader(train_dataset_CNN, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader_CNN  = DataLoader(test_dataset_CNN,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        train_loader_LSTM = DataLoader(train_dataset_LSTM, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader_LSTM  = DataLoader(test_dataset_LSTM,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        ## model is CNN trained on input dataloader  
        model = train_CNN(train_loader=train_loader_CNN, n_classes=n_classes, w_size=window_size, n_epochs=n_epochs_CNN)

        ############################################
        ############### CNN evaluation #############
        ############################################

        ## evaluation of CNN
        test_accuracy, test_precisionn, test_recall = evaluate_model(model_CNN=model, test_loader=test_loader_CNN, n_classes=n_classes)

        #################### ReLU extraction ######################
        ######## training data for Softmax classifier #############
        ###########################################################

        ReLU_train_dataset = ReLU_extractor(model=model, data_loader=train_loader_CNN)

        ###############################################
        ######## Softmax classifier training ##########
        ###############################################
        
        ReLU_train_loader = DataLoader(dataset=ReLU_train_dataset, batch_size=batch_size)

        feature_count = ReLU_train_dataset.tensors[0].shape[1] # the length of one ReLU linearized vector

        model_softmax = train_Softmax(ReLU_train_loader, feature_count, n_classes, n_epochs=n_epochs_softmax)

        ############################################
        ############## CNN-S evaluation ############
        ############################################

        test_accuracy_CNNs, test_precision_CNNs, test_recall_CNNs = evaluate_model(model_CNN=model, test_loader=test_loader_CNN, n_classes=n_classes, softmax=model_softmax)

        print(f"Test accuracy CNNs: {test_accuracy_CNNs}")
        print(f"Test precision CNNs: {test_precision_CNNs}")
        print(f"Test recall CNNS: {test_recall_CNNs}")

        print(f"\nTest accuracy CNN: {test_accuracy}")
        print(f"Test precision CNN: {test_precisionn}")
        print(f"Test recall CNN: {test_recall}")

        accs_CNNs.append(test_accuracy_CNNs)



        lstm_model = train_LSTM(train_loader_LSTM, n_epochs=n_epochs_lstm, n_classes=n_classes)

        # Extract features from last LSTM layer
        X_train_feats, y_train_feats = extract_features(lstm_model, train_loader_LSTM)
        X_test_feats, y_test_feats   = extract_features(lstm_model, test_loader_LSTM)

        # Train Random Forest
        # n_jobs=-1 added for parallel computation
        rf_model = RandomForestClassifier(n_estimators=rf_trees, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_feats, y_train_feats)

        y_pred = rf_model.predict(X_test_feats)

        acc, prec, rec = evaluate_rf(y_test_feats, y_pred)

        print(f"LSTM-RF Test accuracy: {acc}")
        print(f"LSTM-RF Test precision: {prec}")
        print(f"LSTM-RF Test recall: {rec}")


    print(accs_CNNs)

    end_time = time.perf_counter()

    print(f"Time needed for:\n\t{n_fold_CV}-fold crossvalidation\n\t{n_epochs_CNN} epochs of CNN training\n\t{n_epochs_softmax} epochs of SoftMax training\n\n\t is {end_time - start_time} seconds")

