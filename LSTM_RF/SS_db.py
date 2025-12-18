import pandas as pd
import numpy as np
import os
import torch

class protein_sec_struct:

    def __init__(self, id='', sequence='', secondary_structure=list(),
                 pssm=np.empty((0, 20), dtype=int)):

        self.sequence = sequence.strip()
        self.id = id
        self.secondary_structure = list(secondary_structure.strip())
        self.pssm = np.array(pssm, dtype=int)

    # DSSP to Q3 / Q8
    def to_class_transformation(self, n_class=3):
        if int(n_class) == 3:
            for i in range(len(self.secondary_structure)):
                if self.secondary_structure[i] in ['G', 'H', 'I']:
                    self.secondary_structure[i] = 0
                elif self.secondary_structure[i] in ['B', 'E']:
                    self.secondary_structure[i] = 1
                else:
                    self.secondary_structure[i] = 2
        elif int(n_class) == 8:
            for i in range(len(self.secondary_structure)):
                ss = self.secondary_structure[i]
                if ss == 'H':
                    self.secondary_structure[i] = 0
                elif ss == 'G':
                    self.secondary_structure[i] = 1
                elif ss == 'I':
                    self.secondary_structure[i] = 2
                elif ss == 'E':
                    self.secondary_structure[i] = 3
                elif ss == 'B':
                    self.secondary_structure[i] = 4
                elif ss == 'T':
                    self.secondary_structure[i] = 5
                elif ss == 'S':
                    self.secondary_structure[i] = 6
                else:
                    self.secondary_structure[i] = 7

    # PSSM reader
    def read_pssm_fixed(self, path):
        with open(path) as f:
            line = f.readline().rstrip("\n")

        n_cols = len(line) // 3
        widths = [3] * n_cols
        df = pd.read_fwf(path, widths=widths)

        for _, data in df.items():
            self.pssm = np.vstack([self.pssm, np.array(data, dtype=int)])

    # Sliding window function (necessary for LSTM inputs)
    def sliding_window(self, size, index):
        if size % 2 == 0:
            raise ValueError("Window size must be odd")
        start = index - (size - 1) // 2
        end = index + (size - 1) // 2
        # padding at start
        if start < 0:
            pad = np.zeros((abs(start), 20))
            return np.vstack([pad, self.pssm[0:end+1]])
        # padding at end
        if end >= len(self.sequence):
            pad = np.zeros((end - len(self.sequence) + 1, 20))
            return np.vstack([self.pssm[start:len(self.sequence)], pad])
        return self.pssm[start:end+1]

# ss_db
class ss_db:

    def __init__(self, db=list()):
        self.db = list(db)

    def read_db(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        complete_read = False
        for line in lines:
            if line.startswith('>'):
                id = line[1:].strip()
                AAs_line = True
            elif AAs_line:
                sequence = line
                AAs_line = False
                SS_line = True
            elif SS_line:
                secondary_structure = line
                SS_line = False
                complete_read = True
            if complete_read:
                self.db.append(protein_sec_struct(id, sequence, secondary_structure))
                complete_read = False

    def read_pssm_to_db(self):
        for record in self.db:
            file_name = os.path.join("astral_cull_DATA", record.id + ".pssm")

            record.read_pssm_fixed(file_name)

    # Return sliding-window tensors (necessary for LSTM inputs)
    def to_torch_tensor_db(self, window_size, n_classes=3):
        if n_classes not in (3, 8):
            raise Exception("Only 3 or 8 class labelling allowed")

        X = []
        Y = []

        for prot in self.db:
            prot.to_class_transformation(n_class=n_classes)
            for i in range(len(prot.sequence)):
                window = prot.sliding_window(window_size, i)  # (window_size, 20)
                label = prot.secondary_structure[i]
                X.append(window)
                Y.append(label)

        # convert lists to NumPy arrays first, for faster computation
        X_np = np.array(X, dtype=np.float32)  # shape: (num_samples, window_size, 20)
        Y_np = np.array(Y, dtype=np.int64)    # shape: (num_samples,)

        X_tensor = torch.tensor(X_np)
        Y_tensor = torch.tensor(Y_np)
        return X_tensor, Y_tensor
