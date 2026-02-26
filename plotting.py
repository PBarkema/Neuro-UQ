# TO-ReDO
# Invert plots for P*I[invalid] to match Peters convention and for P*I_invalid
# Make sure to redo *I_hat and *U_hat code


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from simulate_data import simulate_data
from data_processing import pool_data_across_blocks, get_smooth_bounds, scale_bounds, get_se
nBlocks = 4
trials_per_block = 60
p = 0.75
p_hat_history, U_hat_history, PE_Uhat_history, P_I_history, invalid_indices, valid_indices, force_reset_point = simulate_data(nBlocks, trials_per_block, p)
PE_Uhats_valid_mean, PE_Uhats_invalid_mean, P_I_valid_mean, P_I_invalid_mean, PE_Uhat_valid_indices, PE_Uhat_invalid_indices, P_I_valid_indices, P_I_invalid_indices = pool_data_across_blocks(PE_Uhat_history, P_I_history, invalid_indices, valid_indices, nBlocks)


# Calculate Means across BLocks
p_hats = np.nanmean(p_hat_history, axis=0)
U_hats= np.nanmean(U_hat_history, axis=0)
PE_Uhats = np.nanmean(PE_Uhat_history, axis=0)
P_I_mean = np.nanmean(P_I_history, axis=0)

# Invert P_I_mean to match Peter's plots except reset trials
P_I_mean[np.setdiff1d(P_I_invalid_indices, [0, trials_per_block+1])] *= -1
P_I_invalid_mean[~np.isin(P_I_invalid_indices, [0, trials_per_block+1])] *= -1

# Calculate SEs across blocks (for shaded error bars)
p_se = get_se(p_hat_history)
U_se = get_se(U_hat_history)
PE_U_se = get_se(PE_Uhat_history)
PE_U_se_valid = get_se(PE_Uhat_history[:, PE_Uhat_valid_indices])
PE_U_se_invalid = get_se(PE_Uhat_history[:, PE_Uhat_invalid_indices])
P_I_se = get_se(P_I_history)
P_I_se_valid = get_se(P_I_history[:, P_I_valid_indices])
P_I_se_invalid = get_se(P_I_history[:, P_I_invalid_indices])


# --- 2. Smoothing, Scaling, and Preparing Data for Plotting ---
trials = np.arange(0, len(p_hats))
blue, orange, green, red = sns.color_palette("deep", n_colors=4)
xlim = -1, len(p_hats)
scaler = MinMaxScaler()
wdw = 4

# A. PE * U (Plot 3)
# 1. Smooth
sm_PE_mean, sm_PE_upper, sm_PE_lower = get_smooth_bounds(PE_Uhats, PE_U_se, wdw)
# 2. Scale
sc_PE_mean, sc_PE_upper, sc_PE_lower, pe_scaler = scale_bounds(sm_PE_mean, sm_PE_upper, sm_PE_lower)
# Raw (Scaled using the same scaler for consistency)
PE_raw_scaled = pe_scaler.transform(PE_Uhats.reshape(-1,1)).flatten()
PE_raw_upper_scaled = pe_scaler.transform((PE_Uhats + PE_U_se).reshape(-1,1)).flatten()
PE_raw_lower_scaled = pe_scaler.transform((PE_Uhats - PE_U_se).reshape(-1,1)).flatten()

# B. PE * U Invalid/Valid (Plot 4)
sm_PE_inv_mean, sm_PE_inv_up, sm_PE_inv_lo = get_smooth_bounds(PE_Uhats_invalid_mean, PE_U_se_invalid, wdw)
sm_PE_val_mean, sm_PE_val_up, sm_PE_val_lo = get_smooth_bounds(PE_Uhats_valid_mean, PE_U_se_valid, wdw)
# Scale these using the same scaler as the main PE*U plot for consistency
sc_PE_inv_mean = pe_scaler.transform(sm_PE_inv_mean.reshape(-1,1)).flatten()
sc_PE_inv_up = pe_scaler.transform(sm_PE_inv_up.reshape(-1,1)).flatten()
sc_PE_inv_lo = pe_scaler.transform(sm_PE_inv_lo.reshape(-1,1)).flatten()
sc_PE_val_mean = pe_scaler.transform(sm_PE_val_mean.reshape(-1,1)).flatten()
sc_PE_val_up = pe_scaler.transform(sm_PE_val_up.reshape(-1,1)).flatten()
sc_PE_val_lo = pe_scaler.transform(sm_PE_val_lo.reshape(-1,1)).flatten()

# C. p_hat * I_hat (Plot 5)
# 1. Smooth
sm_P_mean, sm_P_upper, sm_P_lower = get_smooth_bounds(P_I_mean, P_I_se, wdw)
# 2. Scale
sc_P_mean, sc_P_upper, sc_P_lower, p_scaler = scale_bounds(sm_P_mean, sm_P_upper, sm_P_lower)
# Raw
P_raw_scaled = p_scaler.transform(P_I_mean.reshape(-1,1)).flatten()
P_raw_upper_scaled = p_scaler.transform((P_I_mean + P_I_se).reshape(-1,1)).flatten()
P_raw_lower_scaled = p_scaler.transform((P_I_mean - P_I_se).reshape(-1,1)).flatten()

# D. P * I Invalid/Valid (Plot 6)
sm_P_inv_mean, sm_P_inv_up, sm_P_inv_lo = get_smooth_bounds(P_I_invalid_mean, P_I_se_invalid, wdw)
sm_P_val_mean, sm_P_val_up, sm_P_val_lo = get_smooth_bounds(P_I_valid_mean, P_I_se_valid, wdw)
# Scale these using the same scaler as the main PE*U plot for consistency
sc_P_inv_mean = p_scaler.transform(sm_P_inv_mean.reshape(-1,1)).flatten()
sc_P_inv_up = p_scaler.transform(sm_P_inv_up.reshape(-1,1)).flatten()
sc_P_inv_lo = p_scaler.transform(sm_P_inv_lo.reshape(-1,1)).flatten()
sc_P_val_mean = p_scaler.transform(sm_P_val_mean.reshape(-1,1)).flatten()
sc_P_val_up = p_scaler.transform(sm_P_val_up.reshape(-1,1)).flatten()
sc_P_val_lo = p_scaler.transform(sm_P_val_lo.reshape(-1,1)).flatten()

# E. Difference (Plot 7)
# Note: You calculated difference of smoothed curves.
# diff = smoothed_update_PE - smoothed_update_P.
# Since these were scaled independently, the difference is unitless and specific to the plot.
# We will just calculate SE of the diff history and scale it similarly? 
# Actually, your original code subtracted the SCALED means. 
# To add error bars to (Scaled A - Scaled B), we need the joint variance. 
# Simplification: We will plot the diff of the means, and shade using the diff_se (scaled roughly).
# However, since A and B are scaled differently, calculating precise Error Bars for (Scale(A) - Scale(B)) is mathematically complex.
# ALTERNATIVE: Calculate difference FIRST, then Smooth, then Scale. (Recommended)
# Hippocampus plot
diff_history = PE_Uhat_history + P_I_history
diff_mean = np.nanmean(diff_history, axis=0)
diff_se = get_se(diff_history)
sm_diff_mean, sm_diff_upper, sm_diff_lower = get_smooth_bounds(diff_mean, diff_se, wdw)

# Difference Valid/Invalid SEs
diff_se_invalid = diff_se[P_I_invalid_indices] # Assuming indices align
diff_se_valid = diff_se[P_I_valid_indices]

PE_U_P_I = sc_PE_mean + sc_P_mean # The main line
# Error propogation for (A - B) roughly sqrt(se_a^2 + se_b^2) assuming independence for visualization
combined_se_scaled = np.sqrt((sc_PE_upper - sc_PE_mean)**2 + (sc_P_upper - sc_P_mean)**2)
diff_upper = PE_U_P_I + combined_se_scaled
diff_lower = PE_U_P_I - combined_se_scaled

# F. Difference Valid/Invalid (Plot 8)
# Using raw differences of smoothed values as per original code
diff_inv_mean = sm_PE_inv_mean + sm_P_inv_mean
diff_val_mean = sm_PE_val_mean + sm_P_val_mean
# Approx SE
se_diff_inv = np.sqrt((sm_PE_inv_up - sm_PE_inv_mean)**2 + (sm_P_inv_up - sm_P_inv_mean)**2)
se_diff_val = np.sqrt((sm_PE_val_up - sm_PE_val_mean)**2 + (sm_P_val_up - sm_P_val_mean)**2)

# --- 4. Plotting ---

f, (p_ax, I_ax, PE_ax, PE_V_ax, P_ax, P_V_ax, P_PE_ax, P_PE_V_ax) = plt.subplots(8, 1, sharex=True, figsize=(10, 15))

# Plot 1: p_hat
p_ax.plot(trials, p_hats, c=blue)
p_ax.fill_between(trials, p_hats - p_se, p_hats + p_se, color=blue, alpha=0.2)
p_ax.set_title("$\hat p$", size=12)
p_ax.set(xlim=xlim, ylim=(-.1, 1.1))
p_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')
# Ground Truth
p_ax.plot(trials, [p]*len(trials), c="dimgray", ls="--")

# Plot 2: U_hat
I_ax.plot(trials, U_hats, c=blue)
I_ax.fill_between(trials, U_hats - U_se, U_hats + U_se, color=blue, alpha=0.2)
I_ax.set_title("$\hat U$", size=12)
I_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

# Plot 3: PE * U (Scaled)
PE_ax.plot(trials, sc_PE_mean, c=blue, linewidth=2.5, label='Smoothed Trend')
PE_ax.fill_between(trials, sc_PE_lower, sc_PE_upper, color=blue, alpha=0.2)
# Optional: Plot raw with errors too? Might be too messy. Just raw points or faint line.
PE_ax.plot(trials, PE_raw_scaled, alpha=0.25, c=blue) 
# PE_ax.fill_between(trials, PE_raw_lower_scaled, PE_raw_upper_scaled, color=blue, alpha=0.05) 
PE_ax.set_title(r"PE*$\hat U$", size=12)
PE_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

# Plot 4: PE * U I/V (Smoothed, Not Scaled)
PE_V_ax.plot(PE_Uhat_invalid_indices, sm_PE_inv_mean, c=red) #sm_PE_inv_mean
PE_V_ax.fill_between(PE_Uhat_invalid_indices, sm_PE_inv_lo, sm_PE_inv_up, color=red, alpha=0.2)
PE_V_ax.plot(PE_Uhat_valid_indices, sm_PE_val_mean, c=green)
PE_V_ax.fill_between(PE_Uhat_valid_indices, sm_PE_val_lo, sm_PE_val_up, color=green, alpha=0.2)
PE_V_ax.set_title(r"PE*$\hat U$ I/V", size=12)
PE_V_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

# Plot 5: P * I (Scaled)
P_ax.plot(trials, sc_P_mean, c=blue, linewidth=2.5, label='Smoothed Trend')
P_ax.fill_between(trials, sc_P_lower, sc_P_upper, color=blue, alpha=0.2)
P_ax.plot(trials, P_raw_scaled, alpha=0.25, c=blue)
P_ax.set_title(r"$\hat p$*$\hat I$", size=12)
P_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

# Plot 6: P * I I/V (Smoothed, Not Scaled)
P_V_ax.plot(P_I_invalid_indices, sm_P_inv_mean, c=red)
P_V_ax.fill_between(P_I_invalid_indices, sm_P_inv_lo, sm_P_inv_up, color=red, alpha=0.2)
P_V_ax.plot(P_I_valid_indices, sm_P_val_mean, c=green)
P_V_ax.fill_between(P_I_valid_indices, sm_P_val_lo, sm_P_val_up, color=green, alpha=0.2)
P_V_ax.set_title(r"$\hat p$*$\hat I$ inv/val", size=12)
P_V_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

# Plot 7: Difference (Scaled A - Scaled B)
P_PE_ax.plot(trials, PE_U_P_I, c=blue, linewidth=2.5, label='Smoothed Trend')
P_PE_ax.fill_between(trials, diff_lower, diff_upper, color=blue, alpha=0.2)
P_PE_ax.set_title(r"PE*$\hat U$ + $\hat p$*$\hat I$", size=12)
P_PE_ax.set_xlabel("Trial")
P_PE_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

# Plot 8: Difference I/V
P_PE_V_ax.plot(P_I_invalid_indices, diff_inv_mean, c=red)
P_PE_V_ax.fill_between(P_I_invalid_indices, diff_inv_mean - se_diff_inv, diff_inv_mean + se_diff_inv, color=red, alpha=0.2)
P_PE_V_ax.plot(P_I_valid_indices, diff_val_mean, c=green)
P_PE_V_ax.fill_between(P_I_valid_indices, diff_val_mean - se_diff_val, diff_val_mean + se_diff_val, color=green, alpha=0.2)
P_PE_V_ax.set_title(r"PE*$\hat U$ + $\hat p$*$\hat I$ inv/val", size=12)
P_PE_V_ax.axvline(x=force_reset_point+1, color='r', linestyle='--')

f.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()