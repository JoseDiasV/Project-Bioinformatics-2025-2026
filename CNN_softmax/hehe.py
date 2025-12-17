from tqdm import tqdm, trange
import numpy as np


a=0

for bobik in trange(100000):
    a += bobik

print(a)

# X = np.array()
# Y = np.array()
X = np.empty((4,5,5))
for i in range(4):

    X[i]= np.ones((5,5))
    # if i < 1:
        
    #     print("less than zero")
    # else:
    #     np.append(X,b)
    #     print("more than zero")
    # X = np.append(X,b)

print (X)

print(len('AYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL'))


bb = [1,2,3,4]

print(bb[2:5])