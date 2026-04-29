import numpy as np
import matplotlib.pyplot as plt

def display_all_3d(model):
    n_nodes = model.n_nodes

    # Query vector
    fixed_query_vec = np.zeros((3 * n_nodes))
    cause_idx = 0
    effect_idx = 1
    fixed_query_vec[n_nodes + cause_idx] = 1
    fixed_query_vec[2 * n_nodes + effect_idx] = 1

    # h_reference = torch.zeros(hidden_dim, device=device)  # or some other reference point
    ref_t = 0 # time at which we will compute the nudging effect of the inputs
    ref_q = 0 # query index
    # h_reference = torch.tensor(hidden_trajs_avg_causal[ref_q, :, ref_t], device=device) # we will compute input nudging from a reference trajectory
    # h_reference_proj = pca.transform(hidden_trajs_avg_causal[ref_q, :, ref_t].reshape(1, -1)).squeeze()

    h_reference = torch.tensor(hidden_trajs_avg_noncausal[ref_q, :, ref_t], device=device)
    h_reference_proj = pca.transform(hidden_trajs_avg_noncausal[ref_q, :, ref_t].reshape(1, -1)).squeeze()

    # Compute input directions
    all_input_dirs = []
    for combo in all_input_combos:
        observed_events_vec = np.zeros(n_nodes)
        observed_events_vec[list(combo)] = 1
        input_vec = fixed_query_vec.copy()
        input_vec[:n_nodes] = observed_events_vec
        
        preact = model.Wx(torch.tensor(input_vec, device=device, dtype=torch.float32)) + model.Wh(h_reference)
        h_next = (1 - model.alpha) * h_reference + model.alpha * torch.tanh(preact)
        delta_h = h_next - h_reference  # This is the direction vector
        proj_dir_vec = delta_h.cpu().detach().numpy()
        input_dir = proj_dir_vec @ pca.components_.T
        
        all_input_dirs.append(input_dir)

    init_coords = np.array([np.mean(factor_trajs_avg_causal[:, 0, 0]), np.mean(factor_trajs_avg_causal[:, 1, 0]), np.mean(factor_trajs_avg_causal[:, 2, 0])])

    fig, axes = plt.subplots(
        nrows=n_elev,
        ncols=n_azim,
        figsize=(3.5 * n_azim, 3.5 * n_elev),
        subplot_kw={'projection': '3d'}  # Specify that all subplots are 3D
    )

    # Flatten the axes array for easy iteration, especially if it's 1D
    axes = axes.flatten()
    plot_counter = 0

    print(f"Generating {n_elev * n_azim} plots for different viewing angles...")

    for elev in elevations:
        for azim in azimuths:
            ax = axes[plot_counter]

            # for pair_idx in range(n_pairs):
            for pair_idx in [0]:

                # CAUSAL
                traj_x = factor_trajs_avg_causal[pair_idx, 0, :]
                traj_y = factor_trajs_avg_causal[pair_idx, 1, :]
                traj_z = factor_trajs_avg_causal[pair_idx, 2, :]
                
                # Plot the trajectory line
                ax.plot(traj_x, traj_y, traj_z, lw=1.5, color='k', zorder=0, alpha=0.25)

                # Add markers at intervals (e.g., every 15th point)
                ax.scatter(
                    traj_x[::intertrial_interval],
                    traj_y[::intertrial_interval],
                    traj_z[::intertrial_interval],
                    s=8,
                    marker='o',
                    color=palette[pair_idx],
                    # color='k',
                    zorder=1,
                    alpha=0.25
                )

                ax.text(traj_x[-1], traj_y[-1], traj_z[-1], 'D: %s>%s'%(nodes[cause_idx], nodes[effect_idx]))


                # NON-CAUSAL
                traj_x = factor_trajs_avg_noncausal[pair_idx, 0, :]
                traj_y = factor_trajs_avg_noncausal[pair_idx, 1, :]
                traj_z = factor_trajs_avg_noncausal[pair_idx, 2, :]
                
                # Plot the trajectory line
                ax.plot(traj_x, traj_y, traj_z, lw=1.5, color='lightgray', zorder=0, alpha=0.25)

                # Add markers at intervals (e.g., every 15th point)
                ax.scatter(
                    traj_x[::intertrial_interval],
                    traj_y[::intertrial_interval],
                    traj_z[::intertrial_interval],
                    s=8,
                    marker='^',
                    # color=palette[pair_idx],
                    color='lightgray',
                    zorder=1,
                    alpha=0.25
                )

                ax.text(traj_x[-1], traj_y[-1], traj_z[-1], 'D: %s!>%s'%(nodes[cause_idx], nodes[effect_idx]))
                
            # --- Plot the Input Direction Arrows ---
            for i, input_dir in enumerate(all_input_dirs):
                ax.quiver(
                    h_reference_proj[0], h_reference_proj[1], h_reference_proj[2],                 # Vector origin (x, y, z)
                    input_dir[0], input_dir[1], input_dir[2], # Vector components
                    length=3.0,                   # Sets arrow length in data units
                    normalize=True,               # Scales all arrows to the same size
                    color=dir_cols[i],              # Use a colormap defined earlier
                    arrow_length_ratio=0.15,       # Adjusts arrowhead size
                    lw=1.5,                       # Line width
                    zorder=50                     # Ensure arrows are drawn prominently
                )
                    
            # --- End of Adaptable Plotting Logic ---

            ax.scatter(
                init_coords[0],
                init_coords[1],
                init_coords[2],
                marker='*',
                edgecolor='k',
                facecolor='k',
                s=150,  # Made larger for better 3D visibility
                zorder=10000
            )

            # -- Plot fixed points -- 
            ax.scatter(
                fixed_points_proj[ref_q][q_vals[ref_q]<qv_thresh, 0], fixed_points_proj[ref_q][q_vals[ref_q]<qv_thresh, 1], fixed_points_proj[ref_q][q_vals[ref_q]<qv_thresh, 2],
                marker='x', c=pred[ref_q][q_vals[ref_q]<qv_thresh], cmap='coolwarm', alpha=1
            )

            # Set the viewing angle for the current subplot
            ax.view_init(elev=elev, azim=azim)

            # Set labels and title to identify the perspective
            ax.set_xlabel('PC 1', labelpad=-10)
            ax.set_ylabel('PC 2', labelpad=-10)
            ax.set_zlabel('PC 3', labelpad=-10)
            ax.set_title(f'Elev: {elev}°, Azim: {azim}°', fontsize=10)

            # # Optional: Simplify ticks for a cleaner look in a dense grid
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_zlim([-4, 4])
            
            plot_counter += 1

    # --- 5. Finalize and Show Plot ---
    # If there are any unused subplots (if nrows*ncols > number of plots), hide them.
    for i in range(plot_counter, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Query: %s > %s'%(nodes[cause_idx], nodes[effect_idx]))
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make room for suptitle
    # plt.savefig('../figs/task-optimized-rnn/model080925/avg_factor_trajectories_3d_with_fixedpoints_intrinsic-noise_with-query-inputs_noise-std0.6_with_inputdirs.pdf', format='pdf', bbox_inches='tight', facecolor='w')
    plt.show()