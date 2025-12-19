from tqdm import tqdm, trange
import numpy as np
from main import *
from CNN_pssm import *
from SS_db import *


if __name__ == "__main__":

    window_size = 13
    n_classes = 3
    batch_size = 20
    n_fold_CV = 10
    n_epochs_CNN = 10
    n_epochs_softmax = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


    ################ Data loading ###############

    # data, labels = load_dataset('test_db.fa', window_size)
    data, labels = load_dataset('cullatraldata_one_third.txt', window_size)
    # data, labels = load_dataset('top100lines.fa', window_size)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # keeps class balance
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    ## model is CNN trained on input dataloader  
    model = train_CNN(train_loader=train_loader, n_classes=n_classes, w_size=window_size, n_epochs=n_epochs_CNN)

