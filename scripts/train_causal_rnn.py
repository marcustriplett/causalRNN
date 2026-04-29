from tqdm import tqdm
import argparse
import numpy as np
import causalrnn
from causalrnn import fixed_point_analysis as fpa

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nodes', type=str, help='Comma-separated list', required=True)
	parser.add_argument('--lambda_w', type=float, default=1e-4)
	parser.add_argument('--lambda_h', type=float, default=0.0)
	parser.add_argument('--hold_time', type=int, default=179)
	parser.add_argument('--n_epochs', type=int, default=30000)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--lr', type=float, default=4e-4)
	parser.add_argument('--fp_iters', type=int, default=5000)
	parser.add_argument('--token', type=str, default='')
	parser.add_argument('--savedir', type=str, required=True)

	# Default RNN arguments
	p_spont = 0.2
	p_cause = 0.8
	hidden_dim = 128
	alpha = 0.075
	training_noise_std = 0.025
	n_samples = 75
	intersample_interval = 5
	dynamics = 'pytorch'

	# parse input DAG nodes
	args = parser.parse_args()
	nodes = [str(node) for node in args.nodes.split(',')] 
	hold_time = int(args.hold_time)
	n_epochs = int(args.n_epochs)
	batch_size = int(args.batch_size)

	# ======================= #
	### Train RNN
	# ======================= #

	print('Initializing RNN model')

	model = causalrnn.TanhRNN(
		nodes, 
		noise_std=training_noise_std, 
		hidden_dim=hidden_dim, 
		p_spont=p_spont, 
		p_cause=p_cause,
		dynamics=dynamics
	)

	print('Training model...')

	model.train_model(
		lr=float(args.lr), 
		batch_size=batch_size,
		n_epochs=n_epochs, 
		n_samples=n_samples,
		intersample_interval=intersample_interval, 
		hold_time=hold_time, 
		lambda_w=float(args.lambda_w), 
		lambda_h=float(args.lambda_h)
	)

	print('Model training complete')

	# ======================= #
	### Log PCA
	# ======================= #

	n_experiences_per_dag = 1
	all_dags = causalrnn.dag_tools.enumerate_dags(model.n_nodes)
	n_dags = len(all_dags)

	print('Generating trajectories for PCA...')

	sampled_experiences = np.array([
		[
			causalrnn.dag_tools.sample_experience_from_dag(
				all_dags[dag_idx], 
				hold_time=hold_time, 
				intersample_interval=intersample_interval, n_samples=n_samples
			)
			for _ in range(n_experiences_per_dag)] 
		for dag_idx in range(n_dags)
	])

	seq_len = sampled_experiences.shape[-1]
	sampled_experiences = sampled_experiences.reshape(-1, model.n_nodes, seq_len)

	# Extract hidden trajectories and run PCA
	hidden_trajs_concat = model.get_many_hidden_trajectories_and_concat(sampled_experiences)
	model.do_pca(hidden_trajs_concat)

	print('PCA complete')

	# ======================= #
	### Fixed point analysis
	# ======================= #

	print('Beginning fixed point analysis')

	# Set up multiple FPF seeds from each average trajectory
	max_seed_dags = 25
	sample_times = [15, 100, 370]
	n_experiences = 5
	n_single_trials_to_sample = 2
	fp_seeds = []
	fp_inputs = []
	fp_queries = []

	queries = causalrnn.get_all_queries(seq_len, model.n_nodes, model.all_pairs, model.node_to_idx)
	n_queries = len(queries)

	seed_queries = queries.copy() # We must probe every query

	if n_dags > max_seed_dags:
		print('Total number of possible DAGs is large (%i), sampling %i DAGs at random to seed FPF'%(n_dags, max_seed_dags))

		_seed_dags = causalrnn.dag_tools.seed_dags_true_false_plus_singletons(nodes)
		seed_dags_idxs = np.random.choice(n_dags - len(_seed_dags), max_seed_dags, replace=False)
		seed_dags = _seed_dags + [all_dags[d] for d in seed_dags_idxs]

	else:
		seed_dags = all_dags.copy()

	for dag in tqdm(seed_dags):
		experiences = causalrnn.dag_tools.sample_multiple_experiences_from_dag(dag, n_experiences=n_experiences, intersample_interval=intersample_interval,
			hold_time=hold_time, n_samples=n_samples)
		hidden_trajs, _, _ = model.get_trajectories(experiences)
		avg_hidden_trajs = np.mean(hidden_trajs, axis=0)

		if model.n_nodes <= 3:
			# Sample along some single-trial trajectories if working with a small DAG
			for i in range(n_single_trials_to_sample):
				for q_idx, quer in enumerate(seed_queries):
					for st in sample_times:
						fp_seeds += [hidden_trajs[i, q_idx, :, st]]
						fp_inputs += [np.r_[np.zeros(model.n_nodes), quer[:, 0]]]
						fp_queries += [q_idx]
		
		# Sample along the mean trajectories
		for q_idx, quer in enumerate(seed_queries):
			for st in sample_times:
				fp_seeds += [avg_hidden_trajs[q_idx, :, st]]
				fp_inputs += [np.r_[np.zeros(model.n_nodes), quer[:, 0]]]
				fp_queries += [q_idx]
			
	fp_seeds = np.array(fp_seeds)
	fp_inputs = np.array(fp_inputs)
	fp_queries = np.array(fp_queries)

	print('Total number of FPF seeds:', len(fp_seeds))

	# Begin search for fixed points
	n_possible_fps = fp_inputs.shape[0]
	fixed_point_search = [None for _ in range(n_possible_fps)]
	for n in range(n_possible_fps):
		hstar, converged, qval, qval_hist = fpa.find_fixed_point_fp32(model, fp_inputs[n], fp_seeds[n], lr=1e-3, max_iters=int(args.fp_iters))
		fixed_point_search[n] = {
			'hstar': hstar, 
			'converged': converged, 
			'qval': qval, 
			'qval_hist': qval_hist, 
			'query': fp_queries[n].item(), 
			'h_init': fp_seeds[n]
		}
		print('Converged:', converged, ' qval:', qval)

	model.fixed_point_search = fixed_point_search

	# ======================= #
	### Saving model
	# ======================= #

	print('Saving model...')

	metadat = 'FwdEqn_holdTime%i_epochs%i_batchSize%i_'%(hold_time, n_epochs, batch_size)
	model.save(args.savedir, token=dynamics+metadat+args.token)
	
	print('All tasks complete.')


