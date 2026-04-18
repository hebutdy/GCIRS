from coembedding.function import co_embedding
import numpy as np
from numba import jit
import scanpy as sc
import pandas as pd
import math
import datatable as dt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@jit(nopython=True)
def cal_distance(X):
    adj_mt=np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            #adj_df[i,j]=torch.sqrt(torch.sum(torch.square(X[i,:]-X[j,:])))
            adj_mt[i,j]=np.linalg.norm(X[i,:]-X[j,:])
    return adj_mt

def pseudo_cal_adj(RNA_ret_adata, pseudo_data_adata, outdir, q=20, p=20):
    adata_path = outdir + '/pesudo_co_adata.h5ad'
    if os.path.exists(adata_path):
        print("Data exists, load it")
        adata = sc.read_h5ad(adata_path)
    else:
        adata = co_embedding(data_list=[RNA_ret_adata, pseudo_data_adata], outdir=outdir)
        adata.write(adata_path)
        
    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]   
    
    final_adj_df_path = outdir + '/pesudo_final_adj_df.txt'

    if os.path.exists(final_adj_df_path):
        print("load adj dataframe")
        final_spot_df = pd.read_csv(final_adj_df_path, index_col=0)
    else:
        X = adata.obsm['latent']
        adj_mt = cal_distance(X)
        print("Finish calculate distance")
        
        adj_df = pd.DataFrame(adj_mt, index=adata.obs_names, columns=adata.obs_names)
        adj_df = adj_df.rename_axis('index').reset_index()
        adj_df.index = adata.obs_names

        #find mutual neighbor
        adj_df.loc[sc_data.obs_names,sc_data.obs_names]=np.nan
        new_adj_df=pd.melt(adj_df,id_vars='index',var_name='index2', value_name='dis').dropna(axis=0,how='any')
        new_df1=new_adj_df.sort_values(['index','dis'], ascending=[True, True]).groupby('index', group_keys=False).apply(lambda x: x.head(p))
        new_df2=new_adj_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(p))
        final_df=pd.merge(new_df1, new_df2, how='inner')
        
        spot_spot=final_df[final_df["index2"].isin(list(st_data.obs_names)) & final_df["index"].isin(list(st_data.obs_names))]
        
        spot_cell=final_df[final_df["index2"].isin(list(sc_data.obs_names)) & final_df["index"].isin(list(st_data.obs_names))]
        # spot_cell2=final_df[final_df["index2"].isin(list(st_data.obs_names)) & final_df["index"].isin(list(sc_data.obs_names))]
        
        cell_to_spot_df1 = spot_cell.rename(columns={'index': 'spot1', 'index2': 'cell'})
        cell_to_spot_df2 = spot_cell.rename(columns={'index': 'spot2', 'index2': 'cell'})
        spot_cell_to_spot = cell_to_spot_df1.merge(cell_to_spot_df2,on='cell', suffixes=('_1', '_2'))
        spot_cell_to_spot = spot_cell_to_spot[spot_cell_to_spot['spot1'] != spot_cell_to_spot['spot2']]
        spot_cell_to_spot['dis'] = (spot_cell_to_spot['dis_1'] + spot_cell_to_spot['dis_2']) / 2 

        spot_cell_to_spot = spot_cell_to_spot.groupby(['spot1', 'spot2'], as_index=False)['dis'].mean()
        cell_spot_df = spot_cell_to_spot[['spot1', 'spot2', 'dis']]
        cell_spot_df = cell_spot_df.rename(columns={'spot1': 'index', 'spot2': 'index2'})
                                              
        diff_connections = cell_spot_df[~cell_spot_df.set_index(['index', 'index2']).index.isin(spot_spot.set_index(['index', 'index2']).index)]
        final_spot_df = pd.concat([spot_spot, diff_connections], ignore_index=True)
        
        final_spot_df["dis"] /= final_spot_df["dis"].max()
        final_spot_df["weight"]=1-final_spot_df["dis"]
        final_spot_df.loc[final_spot_df["weight"]==1,["weight"]]=0

    return final_spot_df
