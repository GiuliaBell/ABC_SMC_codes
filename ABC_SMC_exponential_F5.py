# Pyhton code to run the ABC-SMC algorithm to parametrise the exponential model with cell generations
# making use of F5 T cells data.
# Reference: "Approximate Bayesian Computation scheme for parameter inference and model selection
# in dynamical systems" by Toni T. et al. (2008).

# Import the required modules.
import numpy as np
from scipy.linalg import expm

G = 11  # Total number of generations.
dist_gen = 6  # Used to define generation 5+.
n_pars = 4  # Number of parameters in the exponential model.
N0 = 1  # Number of stages in generation 0.
N1 = 1  # Number of stages all generations but 0.

# Reading the data.
data = np.loadtxt("data_F5.txt")
std_dev = np.loadtxt("std_dev_F5.txt")

# Define the time points (unit of hours).
t2 = np.array([72, 96, 120, 144, 168, 240, 288, 432])

# Define the exponential model with cell generations.
def diag(g,N,l,m):
    if g < 5:
        return(np.diag([-(l[g]+m[g])]*N) + np.diag([l[g]]*(N-1),-1))
    else:
        return(np.diag([-(l[5]+m[5])]*N) + np.diag([l[5]]*(N-1),-1))
        
def matrix(N0,N1,l,m):
    M = np.zeros((N0+(G-1)*N1, N0+(G-1)*N1))
    M[0:N0,0:N0] = diag(0,N0,l,m)
    for i in range(1,G):
        M[N0+(i-1)*N1:N0+i*N1,N0+(i-1)*N1:N0+i*N1] = diag(i,N1,l,m)
    M[N0,N0-1] = 2*l[0]
    for i in range(1,G-1):
        if i < 5:
            M[N0+i*N1,N0+i*N1-1] = 2*l[i]
        else:
            M[N0+i*N1,N0+i*N1-1] = 2*l[5]
    return(M)
    
def exp_matrix(N0,N1,inits,times,l,m):
    output = np.zeros((len(times),N0+N1*(G-1)))
    A = matrix(N0,N1,l,m)
    for i in range(len(times)):
        sol = np.dot(expm(A*times[i]),inits)
        output[i] = sol
    return output.T
        
# Define the functions to use in the ABC-SMC algorithm to generate the first epsilon, to run the first iteration
# and to run all the other iterations.

# As it may be difficult to decide on a reasonably large value of epsilon to use at the first iteration,
# we defined the function below to generate it.
def generate_eps1(nn,rr):
    # Empty array to store the distance.
    results = np.empty((0))
    # Empty array to store the accepted parameters.
    params = np.empty((0,n_pars))

    for run in range(nn*rr):
        # Sample the parameters from uniform prior distributions.
        l0, lambd = 10**np.random.uniform(-3,1,2)
        l = np.array([lambd for _ in range(dist_gen)])
        l[0] = l0 
        alpha = 10**np.random.uniform(-5,-1)
        m = np.zeros(dist_gen)
        for i in range(dist_gen):
            m[i] = alpha*i
        C0 = 10**np.random.uniform(4,6)
        inits = np.zeros((N0+(G-1)*N1))
        inits[0] = C0  
        
        # Run the model to compute the expected number of cells in each generation.
        generations = [[] for _ in range(dist_gen)]
        modelexp = exp_matrix(N0,N1,inits,t2,l,m)
        s0 = sum(modelexp[0:N0])
        generations[0].append(s0)
        for i in range(1,dist_gen):
            if i < 5:
                s = sum(modelexp[N0+(i-1)*N1:N0+i*N1])
                generations[i].append(s)
            else:
                s = sum(modelexp[N0+(i-1)*N1:N0+(G-1)*N1])
                generations[i].append(s)
            
        # Compute the distance between the model predictions and the experimental data.
        generationsravel = np.ravel(generations)
        dataravel = np.ravel(data)
        std_ravel = np.ravel(std_dev)
        distance = np.sqrt(np.sum(((generationsravel-dataravel)/std_ravel)**2))
        
        results = np.hstack((results, distance))
        params = np.vstack((params, np.hstack((C0,l0,lambd,alpha))))

    # Compute epsilon to use at the first iteration.
    epsilon = np.median(results)
    return epsilon

# Define the function for the first iteration of ABC-SMC in which the parameters are sampled
# from the uniform prior distributions.
def iteration1(nn):
    # Empty array to store the distance.
    results = np.empty((0,1))
    # Empty array to store the accepted parameters.
    params = np.empty((0,n_pars))
    number = 0  # Counter for the sample size.
    truns = 0  # Counter for the total number of runs.
    while number < nn:
        truns+=1
        # Sample the parameters from uniform prior distributions.
        l0, lambd = 10**np.random.uniform(-3,1,2)
        l = np.array([lambd for _ in range(dist_gen)])
        l[0] = l0 
        alpha = 10**np.random.uniform(-5,-1)
        m = np.zeros(dist_gen)
        for i in range(dist_gen):
            m[i] = alpha*i
        C0 = 10**np.random.uniform(4,6)
        inits = np.zeros((N0+(G-1)*N1))
        inits[0] = C0
        pars=np.hstack((C0,l0,lambd,alpha))

        # Run the model to compute the expected number of cells in each generation.
        generations = [[] for _ in range(dist_gen)]
        modelexp = exp_matrix(N0,N1,inits,t2,l,m)
        s0 = sum(modelexp[0:N0])
        generations[0].append(s0)
        for i in range(1,dist_gen):
            if i < 5:
                s = sum(modelexp[N0+(i-1)*N1:N0+i*N1])
                generations[i].append(s)
            else:
                s = sum(modelexp[N0+(i-1)*N1:N0+(G-1)*N1])
                generations[i].append(s)
            
        # Compute the distance between the model predictions and the experimental data.
        generationsravel = np.ravel(generations)
        dataravel = np.ravel(data)
        std_ravel = np.ravel(std_dev)
        distance = np.sqrt(np.sum(((generationsravel-dataravel)/std_ravel)**2))

        # If the distance is less than epsilon, store the parameters values and increase by one the counter for
        # the sample size.
        if distance < eps1:
            number+=1
            results = np.vstack((results, distance))
            params = np.vstack((params, pars))

    # Compute the weight for each accepted parameter set - at iteration 1, parameter sets have equal weight.
    weights = np.empty((0,1))
    for i in range(nn):
        weights = np.vstack((weights,1/nn))

    # Return the results: distance, accepted parameters, weights and total number of runs.
    return [np.hstack((results,params,weights)), truns]

# Function for the other iterations of the ABC-SMC algorithm, where the parameter values are sampled
# from the posterior distributions of the previous iteration.
def other_iterations(nn,it):
    # Compute uniform areas to sample within in order to perturb the parameters.
    ranges = []
    for i in range(n_pars):
        r1 = np.max(np.log10(ABC_runs[it][:,i+1])) - np.min(np.log10(ABC_runs[it][:,i+1]))
        ranges.append(r1)
    ranges_arr = np.asarray(ranges)
    sigma = 0.1*ranges_arr

    # Define epsilon as median of the accepted distance values from previous iteration.
    epsilon = np.median(ABC_runs[it][:,0])

    # To use when sampling the new parameters.
    p_list = [i for i in range(nn)]

    # Define upper and lower bounds of the prior distributions for each parameter in the model.
    lower_bounds = np.hstack((10**4,10**(-3),10**(-3),10**(-5)))
    upper_bounds = np.hstack((10**6,10,10,10**(-1)))

    # Empty array to store the distance.
    results = np.empty((0))
    # Empty array to store accepted parameters.
    params = np.empty((0,n_pars))
    # Empty array to store the prior samples.
    priors_abc = np.empty((0,n_pars))
    # Empty array to store the weights.
    weights_arr = np.empty((0))

    number = 0 # Counter for the sample size.
    truns = 0 # Counter for the total number of runs.
    while number < nn:
        truns+=1
        check = 0
        # The following while loop is to sample the parameters from the posterior distributions of the previous
        # iteration. Then the parameters are perturbed making use of a uniform perturbation kernel.
        # If the new parameters lie within the initial prior ranges, they are used to obtaining model predictions,
        # otherwise they are sampled again.
        while check < 1:
            # Randomly choose a parameter set from the posterior obtained from the previous iteration.
            choice = np.random.choice(p_list,1,p=ABC_runs[it][:,n_pars+1])
            prior_sample = ABC_runs[it][:,range(1,n_pars+1)][choice]

            # Generate new parameters through perturbation.
            parameters = []                        
            for i in range(n_pars):
                lower = np.log10(prior_sample[0,i])-sigma[i]
                upper = np.log10(prior_sample[0,i])+sigma[i]
                pars = np.random.uniform(lower,upper)
                parameters.append(10**pars)

            # Check that the new parameters lie within the initial prior ranges.
            check_out = 0
            for ik in range(n_pars):
                if parameters[ik] < lower_bounds[ik] or parameters[ik] > upper_bounds[ik]:
                    check_out = 1

            if check_out == 0:
                check+=1

        C0 = float(parameters[0])
        l0, lambd = parameters[1:3]
        l = np.array([lambd for _ in range(dist_gen)])
        l[0] = l0
        m = np.zeros(dist_gen)
        for i in range(dist_gen):
            m[i] = i*parameters[3]
        
        inits = np.zeros((N0+(G-1)*N1))
        inits[0] = C0 
        
        # Run the model to compute the expected number of cells in each generation.
        generations = [[] for _ in range(dist_gen)]
        modelexp = exp_matrix(N0,N1,inits,t2,l,m)
        s0 = sum(modelexp[0:N0]) #it stops at N0-1
        generations[0].append(s0)
        for i in range(1,dist_gen):
            if i < 5:
                s = sum(modelexp[N0+(i-1)*N1:N0+i*N1])
                generations[i].append(s)
            else:
                s = sum(modelexp[N0+(i-1)*N1:N0+(G-1)*N1])
                generations[i].append(s)
                
        # Compute the distance between the model predictions and the experimental data.
        generationsravel = np.ravel(generations)
        dataravel = np.ravel(data)
        std_ravel = np.ravel(std_dev)
        distance = np.sqrt(np.sum(((generationsravel-dataravel)/std_ravel)**2))

        # If the distance is less than epsilon, store the parameters values and increase by one the counter for
        # the sample size.
        if distance < epsilon:
            number+=1
            
            # Compute the weights for the accepted parameter set.
            denom_arr = []
            for j in range(nn):
                weight = ABC_runs[it][j,n_pars+1]
                params_row = ABC_runs[it][j,1:n_pars+1]
                boxs_up = []
                boxs_low = []
                for i in range(n_pars): 
                    boxs_up.append(np.log10(params_row[i]) + sigma[i])
                    boxs_low.append(np.log10(params_row[i]) - sigma[i])  
                outside = 0
                for i in range(n_pars): 
                   if np.log10(parameters[i]) < boxs_low[i] or np.log10(parameters[i]) > boxs_up[i]:
                       outside = 1
                if outside == 1:
                    denom_arr.append(0)
                else:
                    denom_arr.append(weight*np.prod(1/(2*sigma)))

            weight_param = 1/np.sum(denom_arr)
            weights_arr = np.hstack((weights_arr,weight_param))
            
            results = np.hstack((results, distance))
            params = np.vstack((params, parameters))
            priors_abc = np.vstack((priors_abc, prior_sample))

    # Normalise the weights.
    weights_arr2 = weights_arr/np.sum(weights_arr)
    weights_arr3 = np.reshape(weights_arr2, (nn,1))

    # Return the results: distance, accepted parameters, weights and total number of runs.
    return [np.hstack((np.reshape(results,(nn,1)),params,weights_arr3)),epsilon,truns]

# Sample size for the ABC-SMC.
sample_size = 10000
# Number of iterations to run.
num_iters = 7

# To generate the first value of epsilon.
eps1 = generate_eps1(sample_size,1)
Epsilons = [eps1]

# Run the first iteration of ABC-SMC.
first_output = iteration1(sample_size)
ABC_runs = [first_output[0]]

# Run all the other iterations of ABC-SMC.
for iterat in range(num_iters):
   run = other_iterations(sample_size,iterat)
   ABC_runs.append(run[0])
   Epsilons.append(run[1])

# Save the results as a text file.
np.savetxt('Posterior_F5_exponential.txt', ABC_runs[num_iters])