import numpy as np
from scipy.stats import multivariate_normal

from random import randint

# --------------------------------------------------------------
# SHAPE: (n_subjects, n_timepoints, n_rois, n_rois, n_views)
# --------------------------------------------------------------

# Antivectorize given vector

def antiVectorize(vec, m):
    M = np.zeros((m,m))
    M[np.tril_indices(m,k=-1)] = vec
    M= M.transpose()
    M[np.tril_indices(m,k=-1)] = vec
    return M

# Abbreviations: RH = Right Hemisphere, LH = Left Hemisphere
# Data simulation function, using multivariate normal. This method simulates the most realistic dataset we obtained so far.
def multivariate_simulate(n_samples=200,n_time=2,n_views=4):
    # Note that changing the node count is not provided right now, since we use correlation matrix
    # and the mean values of connectivities from real data and it is for 35 nodes.
    # Import all required statistical information.
    allstats = np.load("./stats/REALDATA_LH_AVGMEANS.npy") # Connectivity mean values of LH. You can also try with RH.
    allcorrs = np.load("./stats/REALDATA_LH_AVGCORRS.npy") # Correlation matrix in LH. You can also try with RH.
    all_diffs = np.load("./stats/REAL_TIME_DIFF.npy") # This is an overall representation of time differences in both (LH and RH) datasets.
    times = []
    for t in range(n_time):
        views = []
        for v in range(n_views):
            # Note that we randomly assign a new random state to ensure it will generate a different dataset at each run.
            # Generate data with the correlations and mean values at the current timepoint.
            if t < 2:
                connectomic_means = allstats[t,v]
                data = multivariate_normal.rvs(connectomic_means,allcorrs[t,v],n_samples,random_state=randint(1,9999))
            # If the requested timepoints are more than we have in real data, use the correlation information from the last timepoint.
            else:
                connectomic_means = allstats[-1,v]
                data = multivariate_normal.rvs(connectomic_means,allcorrs[-1,v],n_samples,random_state=randint(1,9999))
            adj = []
            for idx, sample in enumerate(data):
                # Create adjacency matrix.
                matrix = antiVectorize(sample,35)
                # Perturb the real time difference with nonlinear tanh function.
                noise = np.tanh( t / n_time )
                # Further timepoints will have more significant difference from the baseline (t=6 >> t=1).
                matrix = matrix + all_diffs[:,:,v] * ( noise + 1 )
                adj.append(matrix)
            views.append(np.array(adj))
        times.append(np.array(views))
    alldata=np.array(times)
    # Reshape data as: (#n_samples, #n_time, #n_ROIs, #n_ROIs, #n_views)
    alldata = np.transpose(alldata,(2,0,3,4,1))
    print(alldata.shape)
    return alldata

def prepare_data(new_data=False, n_samples=200, n_times=6):
    # Note that data with 200 samples and 6 timepoints is very large (5.8M data points),
    # check your GPU memory to make sure there is enough space to allocate. If not, try:
    # - to reduce dataset size by changing n_samples or n_times.
    # - on CPU (this will allocate memory on RAM) --> This might work for example if you have 1GB GPU memory but 16GB RAM.
    # - on another computer with a better NVIDIA graphics card. --> 2GB GPU memory will not be enough for 5.8M data.
    try:
        if new_data:
            samples = multivariate_simulate(n_samples,n_times)
            np.save('./multivariate_simulation_data.npy',samples)
        else:
            samples = np.load('./multivariate_simulation_data.npy')
    except:
        samples = multivariate_simulate(n_samples,n_times)
        np.save('./multivariate_simulation_data.npy',samples)
    return samples

if __name__ == "__main__":
    X = prepare_data(new_data=True,n_samples=120,n_times=6)