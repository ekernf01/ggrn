import os
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from math import ceil

class SeqRecord():
    def __init__(self,seqid,seq):
        self.id = seqid
        self.seq = seq
        self.GC = GC(seq)
        
class TargetRecord():
    def __init__(self,gene_name,index=None):
        self.gene_name = gene_name
        self.transcription_factors = []
        self.index = index
        self.transcription_factors_indices = []     
    def add_transcription_factor(self,gene_name,index=None):
        self.transcription_factors.append(gene_name)
        self.transcription_factors_indices.append(index)
    def update(self, target_record):
        self.transcription_factors = self.transcription_factors + target_record.transcription_factors
        self.transcription_factors_indices = self.transcription_factors_indices + target_record.transcription_factors_indices  
        

def run_MAGIC(adata, knn = 30, t = 3, use_rep = 'X_pca', neighbors_key = 'neighbors', n_components = 30, 
              metric = 'euclidean'):
    N = adata.shape[0]
    ka_val = int(np.ceil(knn/3))
    nbrs = NearestNeighbors(n_neighbors=int(knn), metric=metric).fit(adata.obsm[use_rep])
    sparse_distance_matrix = nbrs.kneighbors_graph(adata.obsm[use_rep], mode='distance')
    row, col, val = sparse.find(sparse_distance_matrix)
    ka_list = np.asarray([np.sort(val[row == j])[ka_val] for j in np.unique(row)])
    scaled_distance_matrix = sparse.csr_matrix(sparse_distance_matrix/ka_list[:, None])
    x, y, scaled_dists = sparse.find(scaled_distance_matrix)
    W = sparse.csr_matrix((np.exp(-scaled_dists), (x, y)), shape=[N, N])
    W.setdiag(1)
    kernel = W + W.T
    D = np.ravel(kernel.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]
    T = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
    D, V = sparse.linalg.eigs(T, n_components, tol=1e-4, maxiter=1000)
    D, V = np.real(D), np.real(V)
    inds = np.argsort(D)[::-1]
    D, V = D[inds], V[:, inds]
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
    imputed_data_temp = adata.X.copy()
    for steps in range(t):
        imputed_data_temp = T * imputed_data_temp
    try:
        adata.obsm['X_magic'] = imputed_data_temp.toarray()
    except AttributeError:
        adata.obsm['X_magic'] = imputed_data_temp
    
    return adata

# """
# : Progress Bar
# """
# @contextlib.contextmanager
# def tqdm_joblib(tqdm_object):

#     def tqdm_print_progress(self):
#         if self.n_completed_tasks > tqdm_object.n:
#             n_completed = self.n_completed_tasks - tqdm_object.n
#             tqdm_object.update(n=n_completed)

#     original_print_progress = joblib.parallel.Parallel.print_progress
#     joblib.parallel.Parallel.print_progress = tqdm_print_progress

#     try:
#         yield tqdm_object
#     finally:
#         joblib.parallel.Parallel.print_progress = original_print_progress
#         tqdm_object.close()
