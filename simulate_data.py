from __future__ import division
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from modules import optlearner as opt
from importlib import reload
from sklearn.preprocessing import MinMaxScaler

# Simulate multiple blocks of data with environment resets and collect metrics
def simulate_data(nBlocks, block_size, p):
    from importlib import reload
    reload(opt)

    force_reset_point = block_size
    env_resets = 2
    #U_hats_history = np.zeros((nBlocks, block_size *2 +1)) * np.nan
    p_hat_history=np.zeros((nBlocks, block_size *2 +env_resets)) * np.nan
    U_hat_history=np.zeros((nBlocks, block_size *2 +env_resets)) * np.nan
    PE_Uhat_history = np.zeros((nBlocks, block_size *2 +env_resets)) * np.nan
    P_I_history = np.zeros((nBlocks, block_size *2 +env_resets)) * np.nan
    invalid_indices = []#np.zeros((nBlocks, block_size *2 +1)) * np.nan
    valid_indices = []
    for iBlock in range(nBlocks):
        # Initialize
        learner = opt.ProbabilityLearner()
        # Generate Data: 
        # Two environments with .75 predictive accuracy
        rng = np.random.default_rng(iBlock)
    
        data_part1 = rng.binomial(1, p, block_size)
        #data_env_reset = np.random.binomial(1, 0.5, force_reset_p_trials)
        rng = np.random.default_rng(iBlock)
        data_part2 = rng.binomial(1, p, block_size)

        # Manual plot update for reset point (since we want to show the reset trial as a gap in the data, we append NaNs for that trial)

        current_p_dist = learner.pI.sum(axis=1) 
        current_I_dist = learner.pI.sum(axis=0)

        p_hat_reset = np.sum(current_p_dist * learner.p_grid)
        I_hat_reset = np.sum(current_I_dist * learner.I_grid)

        # 2. Append to history
        # We append these values directly to the internal lists
        learner._p_hats.append(p_hat_reset)
        learner._I_hats.append(I_hat_reset)
        learner._data.append(0.5) # NaN ensures no dot is drawn for this fake trial


        learner.fit(data_part1)

        print("Resetting Environment...")
        learner.force_reset_new_tone() 
        # Manual plot update for reset point (since we want to show the reset trial as a gap in the data, we append NaNs for that trial)

        current_p_dist = learner.pI.sum(axis=1) 
        current_I_dist = learner.pI.sum(axis=0)

        p_hat_reset = np.sum(current_p_dist * learner.p_grid)
        I_hat_reset = np.sum(current_I_dist * learner.I_grid)

        # 2. Append to history
        # We append these values directly to the internal lists
        learner._p_hats.append(p_hat_reset)
        learner._I_hats.append(I_hat_reset)
        learner._data.append(0.5) # NaN ensures no dot is drawn for this fake trial

        learner.fit(data_part2)
        # learner.plot_history(ground_truth=p, y=data_part1.tolist() + [0.5] + data_part2.tolist() )
        # plt.axvline(x=force_reset_point, color='r', linestyle='--', label='Environment Reset')
        # plt.legend()
        # plt.show()

        # PB: Using a 0.5 as label projects uncertainty about the environment reset, which is a nice visual touch to show that this trial is different (since the learner has no basis to predict it, it should be at chance)
        y= [0.5] + data_part1.tolist() + [0.5] + data_part2.tolist()
        # Get metrix for this block

        # p_hats, U_hats, PE_Uhats = learner.get_metrics(y=y)

        # DEBUG manually get_metrics
        I_hats = learner.I_hats
        p_hats = learner.p_hats
        #I_hats_invalid = np.where(np.array(learner._data) == 1 & np.array(ground_truth) == 0 || np.array(learner._data) == 0 & np.array(ground_truth) == 1)[0])
        # Convert to arrays once for readability and performance
        l_data = np.array(p_hats)
        gt_data = np.array(y)

        # Does this take into account the 0.5 environment reset?
       # breakpoint()
        invalid_trials = np.where(
            (gt_data < 1) 
        )[0]

        valid_trials = np.where(
            (gt_data > 0 ) 
        )[0]
        #     valid_trials = np.where(
        #     ((gt_data == 1) & (l_data < 0.5)) |
        #     ((gt_data == 1) & (l_data > 0.5))
        # )[0]
        
        
        U_hats = [1/u for u in np.exp(I_hats)]
        U_hats_scaled = (U_hats - np.min(U_hats)) / (np.max(U_hats) - np.min(U_hats))
        U_hats = U_hats_scaled

        PE = (p_hats - y)**2

        # Calculation of PE ^ UU as combined uncertainty measure
        PE_Uhat = PE * U_hats

        I_hats_scaled = (I_hats - np.min(I_hats)) / (np.max(I_hats) - np.min(I_hats))
        I_hats = I_hats_scaled
        P_I = p_hats * I_hats

        # invalid trials for p_I should be inverted
        P_I[invalid_trials] = -1*(p_hats[invalid_trials] * I_hats[invalid_trials])
        P_I[[0, force_reset_point+1]] = 0.5 * I_hats[[0, force_reset_point+1]] # Set reset trials to 0.5 * I_hats to reflect uncertainty

        # Invert P_I_mean to match Peter's plots except reset trials
# P_I_mean[np.setdiff1d(P_I_invalid_indices, [0, trials_per_block+1])] *= -1
# P_I_invalid_mean[~np.isin(P_I_invalid_indices, [0, trials_per_block+1])] *= -1


        #data_history[iBLock,:] = learner._data
        p_hat_history[iBlock,:] = p_hats
        U_hat_history[iBlock,:]= U_hats
        PE_Uhat_history[iBlock,:] = PE_Uhat
        P_I_history[iBlock,:] = P_I
        invalid_indices.append([invalid_trials])
        valid_indices.append([valid_trials])
        learner.reset()
    return p_hat_history, U_hat_history, PE_Uhat_history, P_I_history, invalid_indices, valid_indices, force_reset_point