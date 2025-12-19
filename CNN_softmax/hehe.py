from tqdm import tqdm, trange
import numpy as np
import torch
from main import *
from CNN_pssm import *
from SS_db import *


if __name__ == "__main__":

    # window_size = 13
    n_classes = 8

    # batch_size = 20
    # n_fold_CV = 10
    # n_epochs_CNN = 10
    # n_epochs_softmax = 100
    # device = "cuda" if torch.cuda.is_available() else "cpu"


    # ################ Data loading ###############

    # # data, labels = load_dataset('test_db.fa', window_size)
    # data, labels = load_dataset('cullatraldata_one_third.txt', window_size)
    # # data, labels = load_dataset('top100lines.fa', window_size)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     data,
    #     labels,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=labels  # keeps class balance
    # )

    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset  = TensorDataset(X_test, y_test)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ## model is CNN trained on input dataloader  
    # model = train_CNN(train_loader=train_loader, n_classes=n_classes, w_size=window_size, n_epochs=n_epochs_CNN)

    [torch.tensor([0.5125, 0.0000, 0.0000, 0.4524, 0.0000, 0.2500, 0.5000, 0.3947]), torch.tensor([0.4005, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.3750])]
    [torch.tensor([0.7238, 0.0000, 1.0000, 0.5957, 0.0000, 0.3571, 0.0000, 0.4178]), torch.tensor([0.8462, 0.1429, 0.0000, 0.4821, 0.0000, 0.2436, 0.2174, 0.4000])]
    
    # precs CNNs
    prec_1_CNNs = torch.tensor([0.7238, 0.0000, 1.0000, 0.5957, 0.0000, 0.3571, 0.0000, 0.4178])
    prec_2_CNNs = torch.tensor([0.8462, 0.1429, 0.0000, 0.4821, 0.0000, 0.2436, 0.2174, 0.4000])
    
    prec_1_CNNs = np.asarray(prec_1_CNNs)
    prec_2_CNNs = np.asarray(prec_2_CNNs)

    tmp_obj_CNNs = np.empty((0,n_classes))
    tmp_obj_CNNs= np.vstack([tmp_obj_CNNs, prec_1_CNNs])


    # print(tmp_obj)

    tmp_obj_CNNs= np.vstack([tmp_obj_CNNs, prec_2_CNNs])

    print(tmp_obj_CNNs)

    mean_vect = np.mean(tmp_obj_CNNs, axis=0)

    print(mean_vect, "\n\n\n")

    ## precs_LSTMrf
    precs_LSTMrf = [np.array([0.79651163, 0. , 0.  , 0.56164384, 0. ,0.25      , 0.38888889, 0.38392857]), np.array([0.76595745, 0., 0. , 0.66129032, 0. , 0.26086957, 0.28571429, 0.48039216])]
    
    tmp_obj_LSTM= np.empty((0,n_classes))
    prec_1_LSTMRF = np.array([0.79651163, 0. , 0.  , 0.56164384, 0. ,0.25      , 0.38888889, 0.38392857])
    prec_2_LSTMRF = np.array([0.76595745, 0., 0. , 0.66129032, 0. , 0.26086957, 0.28571429, 0.48039216])

    tmp_obj_LSTM= np.vstack([tmp_obj_LSTM, prec_1_LSTMRF])


    # print(tmp_obj)

    tmp_obj_LSTM= np.vstack([tmp_obj_LSTM, prec_2_LSTMRF])

    print(tmp_obj_LSTM)

    mean_vect = np.mean(tmp_obj_LSTM, axis=0)

    print(mean_vect)



