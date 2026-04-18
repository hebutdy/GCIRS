from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power

def adj_mapping_matrix(adj_df,concat_st_sc,outdir):
    obs_name=list(concat_st_sc.obs_names[concat_st_sc.obs["batch"]=="1"])
    nspot_counts=len(concat_st_sc.obs_names[concat_st_sc.obs["batch"]=="1"])
    adj_matrix=np.zeros((nspot_counts,nspot_counts), dtype=np.float32)
    for i in range(len(adj_df)):
        edge=adj_df.iloc[i,:]
        #print(edge)
        if (edge[0] in obs_name) and (edge[1] in obs_name):
            i=obs_name.index(edge[0])
            j=obs_name.index(edge[1])
            adj_matrix[i,j]=edge[3]
    ssum=np.sum(adj_matrix,1)
    ssum[ssum==0]=1
    D=np.diag(ssum)
    D_hat=fractional_matrix_power(D, -0.5)
    adj_matrix=D_hat@adj_matrix@D_hat
    adj_matrix += np.eye(adj_matrix.shape[0])
    
    return adj_matrix