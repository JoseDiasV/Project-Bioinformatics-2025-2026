import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class protein_sec_struct:

    def __init__(self, id='', sequence='', secondary_structure=list(), pssm = np.empty((0,20), dtype=int)):

        self.sequence = sequence
        self.id = id
        self.secondary_structure = list(secondary_structure)
        self.pssm = np.array(pssm, dtype=int)
    
    def to_class_transformation(self, n_class=3):
        # alters ss string to distinguish only 3 classes of ss (not 8 as output of DSSP)
        if int(n_class) == 3:
            for i in range(len(self.secondary_structure)):

                if self.secondary_structure[i] in ['G', 'H', 'I']:
                    self.secondary_structure[i] = 0 #'H'
                elif self.secondary_structure[i] in ['B', 'E']:
                    self.secondary_structure[i] = 1 #'E'
                else:
                    self.secondary_structure[i] = 2 #'C'
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
                else:  # includes 'C' or ' ' or any non-DSSP char
                    self.secondary_structure[i] = 7

    def read_pssm(self, path): # old reading as csv, its fixed width columns
        
        df = pd.read_csv(path, sep='\s+')
        print(self.id)
        for name, data in df.items():
            #append the data (column in original pssm) into np.array
            
            self.pssm = np.vstack([self.pssm, np.array(data,dtype=int)])

    def read_pssm_fixed(self, path):
        
        with open(path) as f:
            line = f.readline().rstrip("\n")

        n_cols = len(line) // 3

        widths = [3] * n_cols

        print(f'Loading pssm of: {self.id}')

        df = pd.read_fwf(path, widths=widths)
        
        for name, data in df.items():
            #append the data (column in original pssm) into np.array
            
            self.pssm = np.vstack([self.pssm, np.array(data,dtype=int)])

    def sliding_window(self, size, index):

        if size%2 == 0:
            print("Only odd size windows make sense and return symetric surroundings of resiude\n")
            return None
        
        start = int(index - (size-1)/2) # its index
        end = int(index + (size-1)/2) # its index

        if start < 0:
            window = np.vstack([np.zeros((np.abs(start),20)), self.pssm[0:end+1]])
            
            return window
            
        if end >= len(self.sequence)-1:
            overhang = end - (len(self.sequence)-2)
            #print("index",index,"\noverhang",overhang, "\n endind", end, "\n konec proteinu", len(self.sequence)-1)
            window = np.vstack([self.pssm[start:len(self.sequence)+1], np.zeros((overhang,20))])
            return window

        window = self.pssm[start:end+1]

        return window
    
class ss_db:

    def __init__(self, db = list()):
        self.db = list(db)
    
    def read_db(self, path):

        with open(path, 'r') as f:
            lines = f.readlines()

        complete_read = False

        for line in lines:
            if line.startswith('>'):   
                id = line[1:len(line)].strip()
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
                if len(sequence) >= 30:
                    self.db.append(protein_sec_struct(id, sequence, secondary_structure))
                else:
                    print(f'Protein {id} has length of {len(sequence)}, which is less than requiered (30)')
                complete_read = False
    
    def read_pssm_to_db(self): # old one with no exception treatment

        for record in self.db:
            file_name = record.id + ".pssm"

            try:
                record.read_pssm_fixed(file_name)
            except:
                print(f'There was an error loading PSSM of {record.id} REMOVING from dataset')

    def read_pssm_to_db_exc(self):
        valid_records = []

        for record in self.db:
            file_name = record.id + ".pssm"
            try:
                record.read_pssm_fixed(file_name)
                valid_records.append(record)  # keep only valid ones
            except Exception as e:
                print(f'There was an error loading PSSM of {record.id}: {e}')
                print('REMOVING from dataset')

        self.db = valid_records         

    def to_torch_tensor_db(self, window_size, n_classes = 3):

        if n_classes not in (3,8):
            raise Exception("Sorry, only 3 or 8 class labelling is allowed")
        
        n_aas = 0 # number of samples
        for prot in self.db:
            n_aas += len(prot.sequence)
        
        # X = np.empty((n_aas,window_size,20)) # N samples * N aa in the window * 20
        # Y = np.empty((n_aas), dtype=int)

        # listy fungujou pekne
        X = []
        Y = []

        for prot in self.db:
            # define the classes
            prot.to_class_transformation(n_class= n_classes)

            for i in range(min(len(prot.secondary_structure), len(prot.sequence))):
                pssm_window = prot.sliding_window(window_size, i)
                #print(i, '\n', pssm_window)
                label = prot.secondary_structure[i]

                # X[i] = pssm_window
                # Y[i] = label
                #print(f'Label: {label} \n prot_id: {prot.id} \n index: {i}')
                X.append(pssm_window)
                Y.append(label)




        # print(X)
        # print(Y, 'tojevono')     
        # X = list(X)
        # Y = np.array(Y)

        X = np.array(X,dtype=float)


        
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # data, in our case iterable of 13*20 pssm matrix windows
        Y_tensor = torch.tensor(Y, dtype=torch.long) # target, sample labels
        # print(Y, Y_tensor, X_tensor)

        return X_tensor, Y_tensor
        
if __name__ == "__main__":
    dataset = ss_db()

    dataset.read_db('one_prot.fa')

    dataset.read_pssm_to_db()

    # for protein in dataset.db:
    #     print(protein.sliding_window(size=13, index=5))

    
    data, labels = dataset.to_torch_tensor_db(window_size=13)

    dataset = TensorDataset(data, labels)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            
