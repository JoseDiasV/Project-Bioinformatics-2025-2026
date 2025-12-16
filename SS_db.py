import pandas as pd
import numpy as np

class protein_sec_struct:

    def __init__(self, id='', sequence='', secondary_structure=list(), pssm = np.empty((0,20), dtype=int)):

        self.sequence = sequence
        self.id = id
        self.secondary_structure = list(secondary_structure)
        self.pssm = np.array(pssm, dtype=int)
    
    def type_transformation(self):
        # alters ss string to distinguish only 3 classes of ss (not 8 as output of DSSP)
        for i in range(len(self.secondary_structure)):

            if self.secondary_structure[i] in ['G', 'H', 'I']:
                self.secondary_structure[i] = 'H'
            elif self.secondary_structure[i] in ['B', 'E']:
                self.secondary_structure[i] = 'E'
            else:
                self.secondary_structure[i] = 'C'

    def read_pssm(self, path):
        
        df = pd.read_csv(path, delim_whitespace=True)

        for name, data in df.items():
            #append the data (column in original pssm) into np.array
            self.pssm = np.vstack([self.pssm, np.array(data)])

    def sliding_window(self, size, index):

        if size%2 == 0:
            print("Only odd size windows make sense and return symetric surroundings of resiude\n")
            return None
        
        start = int(index - (size-1)/2)
        end = int(index + (size-1)/2)

        if start < 0:
            window = np.vstack([np.zeros((np.abs(start),20)), self.pssm[0:end]])
            
            return window
            
        if end > len(self.sequence):
            overhang = end - (len(self.sequence) - 1)
            window = np.vstack([self.pssm[start:len(self.sequence) - 1], np.zeros((overhang,20))])
            return window

        window = self.pssm[start:end]

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
                self.db.append(protein_sec_struct(id, sequence, secondary_structure))
                complete_read = False
    
    def read_pssm_to_db(self):

        for record in self.db:
            file_name = record.id + ".pssm"

            record.read_pssm(file_name)

if __name__ == "__main__":
    dataset = ss_db()

    dataset.read_db('test_db.fa')

    dataset.read_pssm_to_db()

    for protein in dataset.db:
        print(protein.sliding_window(size=13, index=5))

    # for protein in dataset.db:
    #     protein.type_transformation()
    #     print(protein.secondary_structure)

            
