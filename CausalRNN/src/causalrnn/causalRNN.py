import time
import random
import pickle
import numpy as np
import networkx as nx
from datetime import date, datetime
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian

from . import dag_tools

class TanhRNN(nn.Module):
    def __init__(self, nodes, hidden_dim=128, alpha=0.075, noise_std=0, p_spont=0.2, p_cause=0.8, path='', pca=None,
                 dynamics='pytorch', training_results=None, fixed_point_search=None):
        super().__init__()
        self.alpha = alpha
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.input_dim = self.n_nodes * 3
        self.hidden_dim = hidden_dim

        self.Wx = nn.Linear(self.input_dim, self.hidden_dim)
        self.Wh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, 1)

        self.p_spont = p_spont
        self.p_cause = p_cause
        self.noise_std = noise_std

        self.all_pairs = [(a, b) for a in self.nodes for b in self.nodes if a != b] # all possible edges
        self.n_pairs = len(self.all_pairs)

        self.path = path
        self.pca = pca
        self.training_results = training_results
        self.fixed_point_search = fixed_point_search
        
        assert dynamics in ['bio', 'pytorch']
        self.dynamics = dynamics

        if self.dynamics == 'bio':
            self.update_fn = self._bio_update
            self.output_fn = lambda x, r: r        # logits from r

        elif self.dynamics == 'pytorch':
            self.update_fn = self._pytorch_update
            self.output_fn = lambda x, r: x        # logits from x

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def _bio_update(self, x, u_t):
        preact = self.Wx(u_t) + self.Wh(torch.tanh(x))
        x = (1 - self.alpha) * x + self.alpha * preact
        r = torch.tanh(x)
        return x, r

    def _pytorch_update(self, x, u_t):
        preact = self.Wx(u_t) + self.Wh(x)
        x = (1 - self.alpha) * x + self.alpha * torch.tanh(preact)
        r = torch.tanh(x)        # still keep r for consistency, though not used
        return x, r
    
    def _one_step_state_update(self, x, u_t):
        x_next, _ = self.update_fn(x, u_t)
        return x_next
    
    # Shared forward loop
    def forward(self, u, return_hidden=False):
        seq_len, batch_size, _ = u.shape
        x = torch.zeros(batch_size, self.hidden_dim, device=u.device)
        r = torch.tanh(x)
        hidden_states = []
        logits = torch.zeros((seq_len, batch_size), device=u.device)

        for t in range(seq_len):
            x, r = self.update_fn(x, u[t])

            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std

            if return_hidden:
                hidden_states.append(x.clone())

            logits[t] = self.out(self.output_fn(x, r)).squeeze(1)

        if return_hidden:
            return logits, torch.stack(hidden_states)
        return logits
    
    def train_model(self, lr=1e-4, batch_size=1024, n_epochs=1e4, n_samples=75, intersample_interval=5, 
                    print_every=1000, hold_time=179, lambda_w=1e-4, lambda_h=0):#, fixate_coef=0.1, fixate_time=50, ):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        n_epochs = int(n_epochs)
        losses, accuracies = [], []
        t_start = time.time()

        # Training Loop
        self.train()
        for epoch in range(1, n_epochs + 1):
            x_batch, y_batch = self.generate_batch(batch_size, n_samples=n_samples, 
                                                   intersample_interval=intersample_interval,
                                                   hold_time=hold_time)

            logits, hidden_activity = self(x_batch, return_hidden=True)

            mean_logits = torch.mean(logits[-hold_time:], dim=0)
            loss = criterion(mean_logits, y_batch)

            l2_w_reg = lambda_w * torch.sum(self.Wh.weight ** 2)
            l2_h_reg = lambda_h * torch.mean(hidden_activity ** 2)

            loss += l2_w_reg + l2_h_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(mean_logits) > 0.5).float()
            acc = (preds == y_batch).float().mean().item()
            losses.append(loss.item())
            accuracies.append(acc)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc:.3f}")

        elapsed_time = (time.time() - t_start)/60
        print('Training complete. Elapsed time %.2f min'%elapsed_time)
        self.training_results = {'losses': np.array(losses), 'accuracies': np.array(accuracies), 'elapsed_time': elapsed_time,
                         'lr': lr, 'n_epochs': n_epochs, 'n_samples': n_samples, 'intersample_interval': intersample_interval,
                         'hold_time': hold_time, 'lambda_w': lambda_w, 'lambda_h': lambda_h}#, 'fixate_time': fixate_time, }
        self.hold_time = hold_time
        self.eval()
    
    def jacobians_at(self, h_star, u_star):
        '''
        Compute Jacobians of the one-step state update with respect to both
        hidden state and input, evaluated at (h_star, u_star).

        The local linearization is:
            F(h, u) ~ F(h_star, u_star) + J_recurrent (h - h_star) + J_input (u - u_star)

        '''

        # Convert h_star, u_star, and ensure correct device placement
        if isinstance(h_star, np.ndarray):
            h_star = torch.from_numpy(h_star)
        h_star = h_star.detach().to(self.device).float()
        if h_star.dim() == 2:
            h_star = h_star.squeeze(0)
        assert h_star.dim() == 1, 'h_star must be 1D (hidden_dim,)'

        if isinstance(u_star, np.ndarray):
            u_star = torch.from_numpy(u_star)
        u_star = u_star.detach().to(self.device).float()
        if u_star.dim() == 2:
            u_star = u_star.squeeze(0)
        assert u_star.dim() == 1, 'u_star must be 1D (input_dim,)'

        self.eval() # make sure no noise is added

        # define the state update function (R^H x R^D -> R^H) that jacobian will see
        def state_update_fn(h_vec, u_vec):
            # h_vec: (hidden_dim,), u_vec: (input_dim,)
            x = h_vec.unsqueeze(0)          # (1, hidden_dim)
            u = u_vec.unsqueeze(0)          # (1, input_dim)
            x_next = self._one_step_state_update(x, u)    # (1, hidden_dim)
            return x_next.squeeze(0)        # (hidden_dim,)

        J_h, J_u = jacobian(state_update_fn, (h_star, u_star))

        return J_h, J_u

    def jacobian_at(self, h_star, u_t):
        '''
        Jacobian wrt hidden state only, holding input fixed at u_t
        '''
        J_h, _ = self.jacobians_at(h_star, u_t)
        return J_h

    def do_pca(self, hidden_trajs, n_components=10):
        pca = PCA(n_components=n_components)
        pca.fit(hidden_trajs.T)
        self.pca = pca
        self._pca_flag = True
        return
    
    def project_point(self, _point):
        assert self._pca_flag, 'First run PCA'
        if type(_point) == torch.Tensor:
            point = _point.detach().cpu().numpy()
        else:
            point = _point.copy()

        if len(point.shape) > 1:
            # Multiple points passed to function for projection
            assert point.shape[1] == self.hidden_dim, 'Second dimension should index hidden units'
            proj = self.pca.components_ @ (point.T - self.pca.mean_[:, None])
            return proj.T
        else:
            # Project single point
            return self.pca.components_ @ (point - self.pca.mean_)

    def get_many_hidden_trajectories_and_concat(self, experiences):
        # Construct a bank of hidden trajectories for PCA
        e_shape = experiences.shape
        n_exp, seq_len = e_shape[0], e_shape[-1]
        hidden_trajs = [None for _ in range(self.n_pairs * n_exp)]
        queries = get_all_queries(seq_len, self.n_nodes, self.all_pairs, self.node_to_idx)
        tcount = 0
        with torch.no_grad():
            for e_idx, exp in enumerate(experiences):
                for q_idx, quer in enumerate(queries):
                    q_in = get_input(exp, quer, self.n_nodes, self.device)
                    _, _hidden_traj = self(q_in, return_hidden=True)
                    hidden_traj = _hidden_traj.cpu().detach().numpy().squeeze().T
                    hidden_trajs[tcount] = hidden_traj
                    tcount += 1

        hidden_trajs = np.concatenate(np.array(hidden_trajs), axis=1)
        return hidden_trajs

    def get_trajectories(self, experiences):
        n_experiences = experiences.shape[0]
        hidden_trajs, decision_trajs, factor_trajs = [[None for _ in range(n_experiences)] for _ in range(3)]
        for e_idx, exp in enumerate(experiences):
            hidden_trajs[e_idx], decision_trajs[e_idx], factor_trajs[e_idx] = self._get_trajectories_single_exp(exp)
        hidden_trajs = np.array(hidden_trajs)
        decision_trajs = np.array(decision_trajs)
        factor_trajs = np.array(factor_trajs)
        return hidden_trajs, decision_trajs, factor_trajs

    def _get_trajectories_single_exp(self, experience):
        assert self._pca_flag, 'Run PCA before retrieving trajectories'

        hidden_trajs = [None for _ in range(self.n_pairs)]
        decision_trajs = [None for _ in range(self.n_pairs)]
        factor_trajs = [None for _ in range(self.n_pairs)]
        queries = get_all_queries(experience.shape[1], self.n_nodes, self.all_pairs, self.node_to_idx)
        with torch.no_grad():
            for q_idx, quer in enumerate(queries):
                q_in = get_input(experience, quer, self.n_nodes, self.device)
                q_out, _hidden_traj = self(q_in, return_hidden=True)
                
                hidden_traj = _hidden_traj.cpu().detach().numpy().squeeze().T
                hidden_trajs[q_idx] = hidden_traj
                
                decision_traj = torch.sigmoid(self.out(_hidden_traj.squeeze(1)).squeeze(1))
                decision_trajs[q_idx] = decision_traj.cpu().detach().numpy()

                factor_trajs[q_idx] = self.pca.components_ @ (hidden_trajs[q_idx] - self.pca.mean_[:, None])
            
        factor_trajs = np.array(factor_trajs)
        hidden_trajs = np.array(hidden_trajs)
        decision_trajs = np.array(decision_trajs)
        return hidden_trajs, decision_trajs, factor_trajs
    
    def generate_batch(self, batch_size, n_samples=75, intersample_interval=5, hold_time=179): # fixate_time=50, ):
        '''
        Generates a batch of data using a Noisy-OR model and the compositional query format.
        Input shape: [seq_len, batch_size, 3 * n_nodes]
        '''
        
        n_nodes = self.n_nodes
        seq_len = (n_samples - 1) * intersample_interval + 1
        input_dim = n_nodes * 3
        # full_duration = fixate_time + seq_len + hold_time
        full_duration = seq_len + hold_time
        
        labels = np.zeros(batch_size, dtype=np.float32)
        inputs = np.zeros((full_duration, batch_size, input_dim), dtype=np.float32)

        for b in range(batch_size):
            # Sample a ground truth DAG
            G = dag_tools.sample_random_dag(self.nodes)

            # Sample a query (e.g., ('A', 'B') for "A->B?")
            source_node, target_node = random.choice(self.all_pairs)
            
            # Determine the true label for the query
            labels[b] = 1.0 if G.has_edge(source_node, target_node) else 0.0

            # Simulate node activity for n_samples trials using Noisy-OR
            data = {}
            for node in nx.topological_sort(G):
                parents = list(G.predecessors(node))
                if not parents:
                    # Node without parents
                    p_on = np.full(n_samples, self.p_spont)
                else:
                    # Node has parents
                    parent_activity = np.stack([data[p] for p in parents], axis=1)
                    num_active_parents = parent_activity.sum(axis=1)
                    prob_failure_from_causes = (1 - self.p_cause) ** num_active_parents
                    prob_node_off = (1 - self.p_spont) * prob_failure_from_causes
                    p_on = 1 - prob_node_off
                data[node] = np.random.binomial(1, p_on)

            # Build the full input sequence with the compositional query
            full_input = np.zeros((full_duration, input_dim), dtype=np.float32)
            
            # Create the constant query part of the input
            query_vec = np.zeros(n_nodes * 2, dtype=np.float32)
            source_idx = self.node_to_idx[source_node]
            target_idx = self.node_to_idx[target_node]
            
            # Activate the dimension for the source node in the second block
            query_vec[source_idx] = 1.0

            # Activate the dimension for the target node in the third block
            query_vec[n_nodes + target_idx] = 1.0
            
            # Broadcast the query vector across all time steps
            full_input[:, n_nodes:] = query_vec

            # Add the event data at the specified intervals
            node_activity = np.stack([data[n] for n in self.nodes], axis=1)
            
            for i in range(n_samples):
                t = i * intersample_interval
                if t >= seq_len:
                    break
                # Place data in the first N dimensions
                full_input[t, :n_nodes] = node_activity[i] 
            
            inputs[:, b, :] = full_input

        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
        return inputs, labels
    
    def save(self, savedir, token='', save_time=False):
        assert type(token) == str, 'Token variable must be string.'
        if token == '':
            token = '_'
        else:
            token = '_' + token + '_'
        
        # File name formatting
        _date = str(date.today())
        _time = str(datetime.now().time().hour) + ':' + str(datetime.now().time().minute)
        fn = 'causalRNN_nodes%i'%self.n_nodes
        fn += '_hiddenDim%i'%self.hidden_dim
        fn += '_alpha%.3f'%self.alpha
        fn += '_pSpont%.2f'%self.p_spont
        fn += '_pCause%.2f'%self.p_cause
        fn += '_noiseStd%.3f'%self.noise_std
        fn += token 
        fn += _date
        if save_time:
            fn += '_' + _time
        fn += '.pt'
        
        model_dict = {
            'config': {
                'alpha': self.alpha,
                'nodes': self.nodes,
                'hidden_dim': self.hidden_dim,
                'p_spont': self.p_spont,
                'p_cause': self.p_cause,
                'noise_std': self.noise_std,
                'path': savedir + fn,
                'training_results': self.training_results,
                'fixed_point_search': self.fixed_point_search
            },
            'state_dict': self.state_dict(),
            'pca': pickle.dumps(self.pca)
        }

        torch.save(model_dict, savedir + fn)
        print('Saved model at ', savedir + fn)
        return

def load_model(path):
    model_config = torch.load(path, map_location='cpu', weights_only=False)
    model = TanhRNN(**model_config['config'])
    model.load_state_dict(model_config['state_dict'])
    model.pca = pickle.loads(model_config['pca'])
    if model.pca is not None:
        model._pca_flag = True
    model.eval()
    return model

def get_query(node1, node2, seq_len, node_to_idx, n_nodes=3):
    # Create the constant query part of the input
    query_vec = np.zeros((n_nodes * 2, seq_len), dtype=np.float32)
    source_idx = node_to_idx[node1]
    target_idx = node_to_idx[node2]
    
    # Activate the dimension for the source node in the second block
    query_vec[source_idx] = 1.0

    # Activate the dimension for the target node in the third block
    query_vec[n_nodes + target_idx] = 1.0
    return query_vec
        
def get_all_queries(seq_len, n_nodes, all_pairs, node_to_idx):
    return [get_query(pair[0], pair[1], seq_len, node_to_idx, n_nodes=n_nodes) for pair in all_pairs]

def get_label(dag, pair):
    return dag.has_edge(*pair)

def get_input(experience, query, n_nodes, device):
    return torch.tensor(np.r_[experience, query].T.reshape(experience.shape[1], 1, 3*n_nodes), dtype=torch.float32).to(device)
