import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def pool_data_across_blocks(PE_Uhat_history, P_I_history, invalid_indices, valid_indices, nBlocks):
    
    # 1. Initialize dictionaries to map indices to lists of values
    #    We use lists so we can collect all values for an index and average them later.
    temp_invalid_PE_U = defaultdict(list)
    temp_invalid_P_I = defaultdict(list)
    temp_valid_PE_U = defaultdict(list)
    temp_valid_P_I = defaultdict(list)

    for iBlock in range(nBlocks):
        # Retrieve the metrics for this block
        # PE_Uhat_history is shape (nBlocks, trials)
        block_PE_U_values = PE_Uhat_history[iBlock]
        block_P_I_values = P_I_history[iBlock]
        
        # Retrieve the invalid indices for this block
        # Note: Preserving your logic accessing [0]
        block_invalid_indices = invalid_indices[iBlock][0]
        block_valid_indices = valid_indices[iBlock][0]
        
        # --- INVALID TRIALS ---
        for idx in block_invalid_indices:
            val = block_PE_U_values[idx]
            val2 = block_P_I_values[idx]
            
            # Append value to the list for this specific index
            temp_invalid_PE_U[idx].append(val)
            temp_invalid_P_I[idx].append(val2)

        # --- VALID TRIALS ---
        for idx in block_valid_indices:
            val = block_PE_U_values[idx]
            val2 = block_P_I_values[idx]
            
            # Append value to the list for this specific index
            temp_valid_PE_U[idx].append(val)
            # Adjust to Peters Plots?
            temp_valid_P_I[idx].append(val2)

    # 2. Compute the averages and store them in the final list format
    #    Format: (index_number, average_value)
    pooled_data_invalid_PE_U = [(idx, np.mean(vals)) for idx, vals in sorted(temp_invalid_PE_U.items())]
    pooled_data_invalid_P_I  = [(idx, np.mean(vals)) for idx, vals in sorted(temp_invalid_P_I.items())]
    pooled_data_valid_PE_U   = [(idx, np.mean(vals)) for idx, vals in sorted(temp_valid_PE_U.items())]
    pooled_data_valid_P_I    = [(idx, np.mean(vals)) for idx, vals in sorted(temp_valid_P_I.items())]

    # 2. Sort the pooled data based on the trial index (the first item in the tuple)
    #    This ensures that an error at Trial 5 (from any block) comes before Trial 10
    pooled_data_invalid_PE_U.sort(key=lambda x: x[0])
    pooled_data_valid_PE_U.sort(key=lambda x: x[0])
    pooled_data_invalid_P_I.sort(key=lambda x: x[0])
    pooled_data_valid_P_I.sort(key=lambda x: x[0])


    # 3. Extract just the values into your final list
    PE_Uhat_invalid_combined = [x[1] for x in pooled_data_invalid_PE_U]
    PE_Uhat_invalid_indices= [x[0] for x in pooled_data_invalid_PE_U]

    PE_Uhat_valid_combined = [x[1] for x in pooled_data_valid_PE_U]
    PE_Uhat_valid_indices= [x[0] for x in pooled_data_valid_PE_U]

    # Optional: Convert to numpy array if needed for further math
    PE_Uhat_invalid_combined = np.array(PE_Uhat_invalid_combined)
    PE_Uhat_valid_combined = np.array(PE_Uhat_valid_combined)

    # 3. Extract just the values into your final list
    P_I_invalid_combined = [x[1] for x in pooled_data_invalid_P_I]
    P_I_invalid_indices= [x[0] for x in pooled_data_invalid_P_I]

    P_I_valid_combined = [x[1] for x in pooled_data_valid_P_I]
    P_I_valid_indices= [x[0] for x in pooled_data_valid_P_I]

    # Optional: Convert to numpy array if needed for further math
    P_I_invalid_combined = np.array(P_I_invalid_combined)
    
    P_I_valid_combined = np.array(P_I_valid_combined)

    
    #P_I_invalid_combined = P_I_invalid_combined * -1 # Invert to match Peter's plots


    print(f"Combined {len(PE_Uhat_invalid_combined)} invalid trials from {nBlocks} blocks.")
    print("First 10 values:", PE_Uhat_invalid_combined[:10])
    #PE_Uhats_valid_mean, PE_Uhats_invalid_mean, P_I_valid_mean, P_I_invalid_mean, PE_Uhat_valid_indices, PE_Uhat_invalid_indices, P_I_valid_indices, P_I_invalid_indices
    return PE_Uhat_valid_combined, PE_Uhat_invalid_combined, P_I_valid_combined, P_I_invalid_combined, PE_Uhat_valid_indices, PE_Uhat_invalid_indices,  P_I_valid_indices, P_I_invalid_indices  


# Get smoothed bounds for plotting
def get_smooth_bounds(mean_data, se_data, window):
    """Returns smoothed Mean, Upper (Mean+SE), and Lower (Mean-SE)"""
    upper = mean_data + se_data
    lower = mean_data - se_data
    
    # Smooth all three
    s_mean = pd.Series(mean_data).rolling(window=window, min_periods=1, center=True).mean().values
    s_upper = pd.Series(upper).rolling(window=window, min_periods=1, center=True).mean().values
    s_lower = pd.Series(lower).rolling(window=window, min_periods=1, center=True).mean().values
    
    return s_mean, s_upper, s_lower

# Scales mean and bounds. Fits scaler on mean, transforms bounds.
def scale_bounds(s_mean, s_upper, s_lower, fitted_scaler=None):
    """Scales mean and bounds. Fits scaler on mean, transforms bounds."""
    # Reshape for sklearn
    s_mean_r = s_mean.reshape(-1, 1)
    s_upper_r = s_upper.reshape(-1, 1)
    s_lower_r = s_lower.reshape(-1, 1)
    
    if fitted_scaler is None:
        fitted_scaler = MinMaxScaler()
        s_mean_scaled = fitted_scaler.fit_transform(s_mean_r).flatten()
    else:
        s_mean_scaled = fitted_scaler.transform(s_mean_r).flatten()
        
    s_upper_scaled = fitted_scaler.transform(s_upper_r).flatten()
    s_lower_scaled = fitted_scaler.transform(s_lower_r).flatten()
    
    return s_mean_scaled, s_upper_scaled, s_lower_scaled, fitted_scaler

# Helper to get Standard Error (SE)
def get_se(history_data):
    # Standard Error = Std Dev / sqrt(N)
    return np.nanstd(history_data, axis=0) / np.sqrt(history_data.shape[0])