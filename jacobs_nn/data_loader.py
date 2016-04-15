from bisect import bisect_left
from math import floor
# Base loader that extracts data from files
def load_matrix(fpath, delim = ' ', skip_k = 0):
    mat = []
    with open(fpath, 'r') as fil:
        while skip_k > 0:
            next(fil)
            skip_k -= 1
        mat = [(lin.rstrip()).split(delim) for lin in fil]
    return mat

# Uses load_matrix to get data,
# then use data-specific parameters to
# oraganize data into feature vectors and labels

def load_hw3data(fpath, delim = ',', skip_k = 0):
    mat = load_matrix(fpath, delim, skip_k)
    f_len = len(mat[0])
    xmat = []
    ymat = []
    for i in range(len(mat)):
        try:
            xmat.append( [float(mat[i][j]) for j in range(f_len-1)] )
            ymat.append( mat[i][f_len - 1] )
        except ValueError:
            pass
    return xmat, ymat

def load_occupancy(fpath, delim = ',', skip_k = 1):
    mat = load_matrix(fpath, delim, skip_k)
    f_len = len(mat[0])
    xmat = []
    ymat = []
    for i in range(len(mat)):
        xmat.append( [float(mat[i][j]) for j in range(2, f_len - 1)] )
        ymat.append( mat[i][f_len - 1] )
    return xmat, ymat

def load_sensorless(fpath, delim = ' ', skip_k = 0):
    mat = load_matrix(fpath, delim, skip_k)
    f_len = len(mat[0])
    xmat = []
    ymat = []
    for i in range(len(mat)):
        xmat.append( [float(mat[i][j]) for j in range(f_len - 1)] )
        ymat.append( mat[i][f_len - 1] )
    return xmat, ymat

# feature lists are lists of tuples
# of (word_id, exists?) pairs
# use this for the farm-ad-vect file
def load_farm_vec(fpath, delim = ' ', skip_k = 0):
    mat = load_matrix(fpath, delim, skip_k)
    # takes too long with whole dataset
    mat = mat[:floor(len(mat)/4)]
    xmat = []
    ymat = []
    num_words = 0
    for i in range(len(mat)):
        xmat.append( [tuple([int(n) for n in mat[i][j].split(':')]) \
                        for j in range(1, len(mat[i]))] )
        ymat.append( mat[i][0] )
        for wid, occ in xmat[i]:
            if wid > num_words:
                num_words = wid

    return xmat, ymat, num_words

# feature lists are lists of tuples
# of (word_id, #occurances) pairs
# use this for the farm-ad file
def load_farm(fpath, delim = ' ', skip_k = 0):
    mat = load_matrix(fpath, delim, skip_k)
    # takes too long with whole dataset
    mat = mat[:floor(len(mat)/4)]
    xmat = []
    ymat = []
    num_words = 0
    wid_dict = {}
    for i in range(len(mat)):
        ymat.append( mat[i][0] )
        curr_x = []
        for j in range(1, len(mat[i])):
            wid = None
            word = mat[i][j]
            if mat[i][j] in wid_dict: # word has been seen before
                wid = wid_dict[word]
                word_in_x = False
                for i in range(len(curr_x)):
                    if wid == curr_x[i][0]: # word has been seen in this doc
                        word_in_x = True
                        curr_x = (wid, curr_x[i][1] + 1)
                        break
                if not word_in_x: # word has not been seen in this doc
                    curr_x.append( (wid, 1) )
            else: # word has not been seen before
                num_words += 1
                wid = num_words
                wid_dict.update( {word: wid} ) # add it to dict
                curr_x.append( (wid, 1) ) # add it to feature vector
        xmat.append(sorted(curr_x))
    return xmat, ymat, num_words
