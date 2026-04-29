import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

def find_fixed_point_fp64(rnn, _x_const, _h_init, lr=1e-2, max_iters=2000, tol=1e-5):
    '''
    Find a fixed point of a trained LeakyTanhRNN for constant input x_const.

    Args:
        rnn: trained LeakyTanhRNN
        x_const: tensor [input_dim], constant input (on same device as rnn)
        h_init: tensor [hidden_dim], initial hidden state guess
        lr: learning rate (step size) for optimizer
        max_iters: maximum iterations
        tol: tolerance on residual norm

    Returns:
        h_star: fixed point tensor [hidden_dim]
        converged: bool
        residual: final norm of residual
    '''

    device = rnn.device

    # Extract copies of relevant tensors (fp64)
    Wx = rnn.Wx.weight.detach().to(dtype=torch.float64, device=device)
    Wh = rnn.Wh.weight.detach().to(dtype=torch.float64, device=device)
    bx = rnn.Wx.bias.detach().to(dtype=torch.float64, device=device)

    alpha = float(rnn.alpha)

    # h = h_init.clone().detach().requires_grad_(True)
    h = torch.tensor(_h_init).to(dtype=torch.float64, device=device).clone().requires_grad_(True)
    x_const = torch.tensor(_x_const).to(dtype=torch.float64, device=device)
    optimizer = optim.Adam([h], lr=lr)
    loss_history = []

    for i in tqdm(range(max_iters)):
        optimizer.zero_grad()

        preact = torch.matmul(h, Wh.T) + torch.matmul(x_const, Wx.T) + bx
        h_next = (1 - alpha) * h + alpha * torch.tanh(preact)

        residual = h - h_next
        loss = 0.5 * (residual**2).sum() # q function
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

    converged = loss.item() < tol
    return h.detach(), converged, loss.item(), np.array(loss_history)

def find_fixed_point_fp32(rnn, _x_const, h_init, lr=1e-2, max_iters=2000, tol=1e-5):
    '''
    Find a fixed point of a trained LeakyTanhRNN for constant input x_const.

    Args:
        rnn: trained LeakyTanhRNN for causal judgment
        _x_const: constant input (on same device as rnn)
        h_init: initial hidden state guess
        lr: learning rate (step size) for optimizer
        max_iters: maximum number of iterations
        tol: tolerance on residual norm (determines convergence)

    Returns:
        h_star: tensor of fixed points
        converged: bool
        loss_history: array recording q function at each iteration
    '''

    Wx, Wh, bx = rnn.Wx.weight.to(dtype=torch.float32), rnn.Wh.weight.to(dtype=torch.float32), rnn.Wx.bias.to(dtype=torch.float32)
    h = torch.tensor(h_init).to(dtype=torch.float32, device=rnn.device).clone().detach().requires_grad_(True)
    x_const = torch.tensor(_x_const).to(dtype=torch.float32, device=rnn.device)
    optimizer = optim.Adam([h], lr=lr)    
    alpha = float(rnn.alpha)

    loss_history = []

    for i in tqdm(range(max_iters)):
        optimizer.zero_grad()

        preact = torch.matmul(h, Wh.T) + torch.matmul(x_const, Wx.T) + bx #
        h_next = (1 - alpha) * h + alpha * torch.tanh(preact) #

        residual = h - h_next
        loss = 0.5 * (residual**2).sum() # q function
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

    converged = loss.item() < tol
    return h.detach(), converged, loss.item(), np.array(loss_history)

def merge_fixed_points(fixed_point_dicts, merge_threshold=1e-1, q_threshold=1e-3):
    fp_states = np.stack(
        [fp['hstar'].detach().cpu().numpy() for fp in fixed_point_dicts],
        axis=0
    )  # shape: (N, hidden_dim)

    fp_queries = np.array([fp['query'] for fp in fixed_point_dicts])
    fp_qvals = np.array([fp['qval'] for fp in fixed_point_dicts])

    # Cluster within each query separately
    rep_indices = []   # indices of representative FPs (in original list)
    clusters = []      # list of lists of indices (cluster memberships)
    cluster_queries = []  # which query each cluster corresponds to

    for qval_query in np.unique(fp_queries):
        # indices of FPs with this query context
        idxs = np.where(fp_queries == qval_query)[0]
        if idxs.size == 0:
            continue

        # sort those indices by qval (ascending = slowest / best first)
        local_order = idxs[np.argsort(fp_qvals[idxs])]

        rep_indices_q = []   # reps within this query
        clusters_q = []      # clusters within this query

        for idx in local_order:
            qv = fp_qvals[idx]
            if qv >= q_threshold:
                continue

            this_fp = fp_states[idx]

            if len(rep_indices_q) == 0:
                # first representative for this query
                rep_indices_q.append(idx)
                clusters_q.append([idx])
                continue

            # distances to existing representatives (same query by construction)
            dists = np.array([
                np.linalg.norm(this_fp - fp_states[j])
                for j in rep_indices_q
            ])

            if np.any(dists < merge_threshold):
                # belongs to an existing cluster; keep earlier rep (slower q)
                nearest = int(np.argmin(dists))
                clusters_q[nearest].append(idx)
            else:
                # new cluster within this query
                rep_indices_q.append(idx)
                clusters_q.append([idx])

        # append this query’s clusters to global lists
        rep_indices.extend(rep_indices_q)
        clusters.extend(clusters_q)
        cluster_queries.extend([qval_query] * len(rep_indices_q))

    # Final merged FPs (one representative per cluster)
    fixed_points_merged = [fixed_point_dicts[i] for i in rep_indices]
    fixed_points_merged_queries = np.array([fp['query'] for fp in fixed_points_merged])
    # fixed_points_merged_proj = np.array([model.project_point(fp['hstar']) for fp in fixed_points_merged])
    return fixed_points_merged, fixed_points_merged_queries