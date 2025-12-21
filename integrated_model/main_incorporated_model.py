from header import load_dataset, train_CNN, train_Softmax, ReLU_extractor, evaluate_model, train_LSTM, extract_features, evaluate_rf, ensemble_prediction

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import numpy as np

import time

if __name__ == "__main__":

    start_time = time.perf_counter()

    # AMP (Automatic Mixed Precision) On/Off flag, use =torch.cuda.is_available() for both ON and 
    # automatically OFF when in use on a non-GPU-supported machine, use =False for OFF
    # drastically improves running times when used
    use_amp = torch.cuda.is_available()
    if use_amp:
        # usually already enabled for TF32 suported GPUs, specified for locking behaviour purposes
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    window_size = 13
    n_classes = 8
    batch_size = 20
    n_fold_CV = 3 # should be 10
    n_epochs_CNN = 2 # should be 10
    n_epochs_softmax = 10 # should be 10
    n_epochs_lstm = 2 # should  be 9
    rf_trees = 500
    path_db = 'cullatraldata_one_seventh.txt'
    #path_db = 'test_db_modified.fa'

    device = "cuda" if torch.cuda.is_available() else "cpu"


    ################ Data loading ###############

    data, labels = load_dataset(path_db, window_size, n_classes=n_classes)

    # input format for CNN is (batchsize, 1, W, 20) for LSTM (batchsize, W, 20)
    data_CNN = data.unsqueeze(1)

    ## data indexes K-Fold splitting ##
    skf = StratifiedKFold( # keeps class balance
            n_splits=n_fold_CV,
            shuffle=True,
            random_state=42
        )
    

    ## for results storage ##
    accs_CNNs = []
    accs_CNN = []
    accs_LSTM_rf = []
    accs_ens = []

    precs_CNNs = np.empty((0,n_classes))
    precs_CNN = np.empty((0,n_classes))
    precs_LSTM_rf = np.empty((0,n_classes))
    precs_ens = np.empty((0,n_classes))
    
    
    # repeat for cross validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):

        print(f"\n===== Fold {fold + 1}/{n_fold_CV} =====")

        ####### Dataset preparation ########
        ####################################

        ## index-based data-splits construction ## 
        X_CNN_train = data_CNN[train_idx]
        X_LSTM_train = data[train_idx]
        y_train = labels[train_idx]

        X_CNN_test  = data_CNN[test_idx]
        X_LSTM_test = data[test_idx]
        y_test  = labels[test_idx]

        ## prepare test and train TensorDataset and DataLoader for CNN ##
        train_dataset_CNN = TensorDataset(X_CNN_train, y_train)
        test_dataset_CNN  = TensorDataset(X_CNN_test, y_test)

        train_loader_CNN = DataLoader(train_dataset_CNN, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, persistent_workers=False)
        test_loader_CNN  = DataLoader(test_dataset_CNN,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=False)

        ## prepare test and train dataset and loader for LSTM ##
        train_dataset_LSTM = TensorDataset(X_LSTM_train, y_train)
        test_dataset_LSTM  = TensorDataset(X_LSTM_test, y_test)

        train_loader_LSTM = DataLoader(train_dataset_LSTM, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, persistent_workers=False)
        test_loader_LSTM  = DataLoader(test_dataset_LSTM,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=False)


        ############################################
        #############  CNN training    #############
        ############################################
        ## model is CNN trained on input dataloader 
        # 
        model = train_CNN(train_loader=train_loader_CNN, n_classes=n_classes, w_size=window_size, n_epochs=n_epochs_CNN, use_amp=use_amp)

        ############################################
        ############### CNN evaluation #############
        ############################################

        ## evaluation of CNN
        CNN_prob_preds, test_accuracy, test_precisionn, test_recall = evaluate_model(model_CNN=model, test_loader=test_loader_CNN, n_classes=n_classes)

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

        CNNs_prob_preds, test_accuracy_CNNs, test_precision_CNNs, test_recall_CNNs = evaluate_model(model_CNN=model, test_loader=test_loader_CNN, n_classes=n_classes, softmax=model_softmax)

        ## update ##
        accs_CNNs.append(test_accuracy_CNNs)
        accs_CNN.append(test_accuracy)

        precs_CNNs = np.vstack([precs_CNNs, test_precision_CNNs])
        precs_CNN = np.vstack([precs_CNN, test_precisionn])


        ############################################
        ############### LSTM training  #############
        ############################################
        lstm_model = train_LSTM(train_loader_LSTM, n_epochs=n_epochs_lstm, n_classes=n_classes, use_amp=use_amp)

        
        ############## extract features ############
        ##### training data for RF classifier ######
        ############################################
        X_train_feats, y_train_feats = extract_features(lstm_model, train_loader_LSTM)
        X_test_feats, y_test_feats   = extract_features(lstm_model, test_loader_LSTM)


        ############################################
        ################ RF training ###############
        ############################################

        # n_jobs=-1 added for parallel computation
        rf_model = RandomForestClassifier(n_estimators=rf_trees, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_feats, y_train_feats)


        ############################################
        ########### LSTM RF evaluation #############
        ############################################

        y_pred = rf_model.predict(X_test_feats)
        acc, prec, rec = evaluate_rf(y_test_feats, y_pred, n_classes=n_classes)

        ## update ##
        accs_LSTM_rf.append(acc)
        precs_LSTM_rf = np.vstack([precs_LSTM_rf, prec])


        ###################################
        ######## ENSEMBLE prediction ######
        ########## and evaluation ######### 
        ###################################

        ## predicted probabilities from LSTM-RF
        LSTMrf_prob_preds = rf_model.predict_proba(X_test_feats)

        ensemble_prob_preds = ensemble_prediction(pred_CNNs=CNNs_prob_preds, pred_LSTMrf=LSTMrf_prob_preds)
        ensemble_hard_preds = np.argmax(ensemble_prob_preds, axis=1)

        acc_ens, prec_ens, rec_ens = evaluate_rf(y_test_feats, ensemble_hard_preds, n_classes=n_classes)

        ## update ##
        accs_ens.append(acc_ens)
        precs_ens = np.vstack([precs_ens, prec_ens])

        # free VRAM after each fold, including deleting models and intermediate variables, which are no longer needed
        del model, lstm_model, ReLU_train_dataset, X_train_feats, X_test_feats, y_train_feats, y_test_feats
        torch.cuda.empty_cache()
    
    end_time = time.perf_counter()

    ############################################
    ########### LSTM RF evaluation #############
    ############################################

    mean_precs_CNN = np.mean(precs_CNN, axis=0)
    mean_precs_CNNs = np.mean(precs_CNNs, axis=0)
    mean_precs_LSTMrf = np.mean(precs_LSTM_rf, axis=0)
    mean_precs_ens= np.mean(precs_ens, axis=0)

    print(f"\n===== Cross-Validation Q{n_classes} Results =====\n")

    for i, (cnn_s, lstm_rf, cnn, ens) in enumerate(zip(accs_CNNs, accs_LSTM_rf, accs_CNN, accs_ens), start=1):
        print(
            f"Fold {i:2d} | "
            f"CNN+Softmax: {cnn_s:.4f} | "
            f"LSTM-RF: {lstm_rf:.4f} | "
            f"CNN: {cnn:.4f} | "
            f"EN-CSLR: {ens:.4f}"
        )

    print(f"\n===== Average Q{n_classes} over Folds =====\n")

    print(f"Avg CNN+Softmax : {np.mean(accs_CNNs):.4f}")
    print(f"Avg LSTM-RF     : {np.mean(accs_LSTM_rf):.4f}")
    print(f"Avg CNN         : {np.mean(accs_CNN):.4f}")
    print(f"Avg EN-CSLR     : {np.mean(accs_ens):.4f}\n")


    if n_classes == 3:
        label_levels = ['H', 'E', 'C']
    else:
        label_levels = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'C']


    print(f"\n===== Average per-class precision over Folds =====\n")

    for (label, cnn_s, cnn, lstm_rf, ens) in zip(label_levels, mean_precs_CNNs, mean_precs_CNN, mean_precs_LSTMrf, mean_precs_ens):
        print(
            f"Q_{label} | "
            f"CNN+Softmax: {cnn_s:.4f} | "
            f"LSTM-RF: {lstm_rf:.4f} | "
            f"CNN: {cnn:.4f} | "
            f"EN-CSLR: {ens:.4f}"
        )

    print(f"""
    Time needed for:\n
    \t{n_fold_CV}-fold crossvalidation\n
    \t{n_epochs_CNN} epochs of CNN training\n
    \t{n_epochs_lstm} epochs of LSTM training\n
    \t{n_epochs_softmax} epochs of SoftMax training\n\n
    is {end_time - start_time} seconds
    """)
