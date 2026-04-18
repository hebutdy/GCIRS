import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import softmax
from tool.deconv_metric import CalDataMetric
from tool.config import opt
from model.layers import GraphConvolution
from tool.earlystopping import EarlyStopping
from model import my_losses


class my_coGCN(nn.Module):
    def __init__(self,
                 in_feat,
                 hidden_dims=[1024, 512],
                 out_feat=128,
                 dropout=0.5,
                 use_bn=True,
                 activation='leaky_relu'):
        super(my_coGCN, self).__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = []
        self.dropout = nn.Dropout(p=dropout)
        self.use_bn = use_bn

        dims = [in_feat] + hidden_dims + [out_feat]

        for i in range(len(dims) - 1):
            self.layers.append(GraphConvolution(dims[i], dims[i + 1]))

            if activation == 'relu':
                self.acts.append(nn.ReLU())
            elif activation == 'leaky_relu':
                self.acts.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if use_bn and i < len(dims) - 2:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if self.use_bn and i < len(self.bns):
                x = self.bns[i](x)
            if i < len(self.layers) - 1:
                x = self.acts[i](x)
                x = self.dropout(x)
        return x


class SharedGraphModel(nn.Module):
    def __init__(self,
                 in_feat_shared,
                 n_celltype,
                 gcn_hidden_dims=[1024, 512],
                 dim_shared=128,
                 decoder_hidden=128,
                 proj_dim=64,
                 dropout=0.5,
                 use_projector=True,
                 use_bn=True,
                 activation='leaky_relu'):
        super().__init__()
        self.use_projector = use_projector

        self.gcn_shared = my_coGCN(
            in_feat=in_feat_shared,
            hidden_dims=gcn_hidden_dims,
            out_feat=dim_shared,
            dropout=dropout,
            use_bn=use_bn,
            activation=activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_shared, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, n_celltype)
        )

        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(dim_shared, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )

    def forward(self, x_shared, adj):
        h_shared = self.gcn_shared(x_shared, adj)
        pred = torch.log_softmax(self.decoder(h_shared), dim=1)
        return pred, h_shared

    def info_nce_loss(self, z1, z2, temperature=0.1):
        if not self.use_projector:
            raise ValueError("Projector disabled")
        z1 = F.normalize(self.projector(z1), dim=1)
        z2 = F.normalize(self.projector(z2), dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(sim_matrix, labels)


def coGCN_training(concat_st_sc,
                   concat_pseudo_sc,
                   adj, pseudo_adj,
                   gd_results, pseudo_results,
                   n_celltype, α, β,
                   device,
                   gcn_hidden_dims=[1024, 512],
                   dim_shared=128,
                   decoder_hidden=128,
                   proj_dim=64,
                   dropout=0.5,
                   use_projector=True,
                   use_bn=True,
                   activation='leaky_relu'):

    in_feat_shared = concat_st_sc.shape[1]
    model = SharedGraphModel(
        in_feat_shared=in_feat_shared,
        n_celltype=n_celltype,
        gcn_hidden_dims=gcn_hidden_dims,
        dim_shared=dim_shared,
        decoder_hidden=decoder_hidden,
        proj_dim=proj_dim,
        dropout=dropout,
        use_projector=use_projector,
        use_bn=use_bn,
        activation=activation
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_LR[0], gamma=opt.decay_LR[1])

    st_data = concat_st_sc[concat_st_sc.obs['batch'] == "1"]
    pseudo_data = concat_pseudo_sc[concat_pseudo_sc.obs['batch'] == "1"]

    st_input = torch.FloatTensor(st_data.X.A).to(device)
    pseudo_input = torch.FloatTensor(pseudo_data.X.A).to(device)
    adj = torch.FloatTensor(adj).to(device)
    pseudo_adj = torch.FloatTensor(pseudo_adj).to(device)

    pseudo_results_tensor = torch.FloatTensor(np.array(pseudo_results, dtype=np.float32)).to(device)
    st_results_array = np.array(gd_results, dtype=np.float32)

    loss_fn = nn.KLDivLoss(reduction='batchmean')
    max_pcc = 0.0
    max_ssim = 0.0
    max_jsd = 1.0
    max_rmse = 1.0
    best_M = None
    early_stopping = EarlyStopping(patience=80, verbose=True)

    for epoch in range(opt.max_epoch):
        model.train()
        pred_st, shared_st = model(st_input, adj)
        pred_pseudo, shared_pseudo = model(pseudo_input, pseudo_adj)

        pred_st_normalized = torch.nan_to_num((pred_st.T / pred_st.sum(dim=1)).T, nan=0.0)
        pred_pseudo_normalized = torch.nan_to_num((pred_pseudo.T / pred_pseudo.sum(dim=1)).T, nan=0.0)
        log_pred_pseudo = torch.log_softmax(pred_pseudo_normalized, dim=1)

        loss_kl = loss_fn(log_pred_pseudo, pseudo_results_tensor)

        min_nodes = min(shared_st.size(0), shared_pseudo.size(0))
        loss_struct = model.info_nce_loss(shared_st[:min_nodes], shared_pseudo[:min_nodes])

        total_loss = α * loss_struct + β * loss_kl

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            M_probs1 = pred_st_normalized.detach().cpu().numpy()
            M_probs1 = pd.DataFrame(M_probs1,
                                    index=concat_st_sc.obs_names[n_celltype:],
                                    columns=concat_st_sc.obs_names[0:n_celltype])
            M_probs1 = M_probs1.loc[:, np.unique(M_probs1.columns)].fillna(0).values
            loss_val = my_losses.val_loss(M_probs1, st_results_array)
            if loss_val[0] > max_pcc:
                max_pcc = loss_val[0]
                # best_M = pred_st_normalized
            if loss_val[2] < max_jsd:
                max_jsd = loss_val[2]
                best_M = pred_st_normalized
            if loss_val[1] > max_ssim:
                max_ssim = loss_val[1]
                # best_M = pred_st_normalized
            if loss_val[3] < max_rmse:
                max_rmse = loss_val[3]
                # best_M = pred_st_normalized
                
                

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}: Total Loss {total_loss.item():.4f}")
            print(f"Struct Contrast Loss: {loss_struct.item():.4f}, KL Loss: {loss_kl.item():.4f}")
            print(loss_val)
            print(f"Max PCC: {max_pcc:.4f}")
            print(f"Max SSIM: {max_ssim:.4f}")
            print(f"Max JSD: {max_jsd:.4f}")
            print(f"Max RMSE: {max_rmse:.4f}")

        scheduler.step()
        early_stopping(total_loss.item(), model)
        if early_stopping.early_stop:
            print(f"EarlyStopping at epoch {epoch + 1}")
            break

    with torch.no_grad():
        return best_M.cpu().numpy()


def start(concat_st_sc,
          concat_pseudo_sc,
          adj, pseudo_adj,
          gd_results, pseudo_results,
          α, β,
          device='cpu', seed=2022,
          gcn_hidden_dims=[1024, 512],
          dim_shared=128,
          decoder_hidden=128,
          proj_dim=64,
          dropout=0.5,
          use_projector=True,
          use_bn=True,
          activation='leaky_relu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(device)
    n_celltype = len(concat_st_sc.obs[concat_st_sc.obs['batch'] == '0'])

    output = coGCN_training(
        concat_st_sc,
        concat_pseudo_sc,
        adj, pseudo_adj,
        gd_results, pseudo_results,
        n_celltype, α, β, device,
        gcn_hidden_dims, dim_shared,
        decoder_hidden, proj_dim,
        dropout, use_projector,
        use_bn, activation
    )

    adata_map = sc.AnnData(
        X=output,
        obs=concat_st_sc.obs[n_celltype:].copy(),
        var=concat_st_sc.obs[0:n_celltype].copy(),
    )
    return adata_map
