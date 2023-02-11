import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist

class DataGenerator:
    def __init__(self, T):
        self.T = T

class DataGeneratorSBM:
    def __init__(self, T):
        """Data Generator for the Stochastic Block Model (SBM)
        Args:
        T (int): the time length for generated data
        """
        self.T = T
        
    def __generate_initial(self, K, N, a, b, seed):
        # Z
        alpha = jnp.ones(K)
        pi = dist.Dirichlet(alpha).sample(random.PRNGKey(seed))
        Z = dist.Multinomial(probs=pi).sample(random.PRNGKey(seed), sample_shape=(N,))
        
        # X
        eta = dist.Beta(a, b).sample(random.PRNGKey(seed), sample_shape=(K, K))
        
        p = Z @ eta @ Z.T

        with numpyro.plate('rows', N):
            with numpyro.plate('cols', N):
                X = numpyro.sample(
                    'X',
                    dist.Bernoulli(probs=p), 
                    rng_key=random.PRNGKey(seed)
                )
        
        return X, Z, pi, eta
    
    
    def __generate_data_sequential_randomtrans(self, X, Z, eta=None, ratio=0.1, n_max=10, seed=0):
        key = random.PRNGKey(seed)

        N = X.shape[0]
        K = Z.shape[1]

        Z_new = Z.copy()
        X_new = X.copy()
        for k in range(K):
            for l in range(K):
                point = Z[:, k].reshape(-1, 1).dot(Z[:, l].reshape(1, -1 )).astype(bool)
                point_list = jnp.where(point)
                n = len(point_list[0])
                n_change = min([int(ratio * n), n_max])
                idxes = random.choice(key, jnp.arange(n), shape=(n_change,))
                X_new = X_new.at[point_list[0][idxes], point_list[1][idxes]].set(
                            dist.Bernoulli(probs=eta[k, l]).sample(
                                key, sample_shape=(n_change,))
                        )
        
        return X_new, Z_new
    
    
    def __generate_data_abrupt_pi(self, X, Z, eta=None, n_trans=None, idx_before=None, idx_after=-1, seed=0):
        key = random.PRNGKey(seed)

        N = X.shape[0]
        K = Z.shape[1]

        Z_new = Z.copy()
        idx_change = random.choice(key, jnp.where(Z[:, idx_before] == 1)[0], shape=(n_trans, ))
        Z_new = Z_new.at[idx_change, idx_before].set(0)
        Z_new = Z_new.at[idx_change, idx_after].set(1)

        X_new = X.copy()

        for i in idx_change:
            idx_i = idx_after
            for j in range(N):
                idx_j = jnp.where(Z_new[j, :] == 1)[0]
                X_new = X_new.at[i, j].set(
                            dist.Bernoulli(probs=eta[idx_i, idx_j]).sample(
                                key, sample_shape=(1,))[0][0]
                )
                X_new = X_new.at[j, i].set(
                            dist.Bernoulli(probs=eta[idx_j, idx_i]).sample(
                                key, sample_shape=(1,))[0][0]
                )

        return X_new, Z_new

    
    def __generate_data_abrupt_eta(self, X, Z, eta, seed=0):
        key = random.PRNGKey(seed)

        N = X.shape[0]
        K = Z.shape[1]

        X_new = jnp.zeros((N, N), dtype=int)

        for k in range(K):
            for l in range(K):
                point = Z[:, k].reshape(-1, 1).dot(Z[:, l].reshape(1, -1 )).astype(bool)
                if jnp.sum(point) > 0:
                    X_new = X_new.at[point].set(dist.Bernoulli(probs=eta[k, l]).sample(
                                                key, sample_shape=(jnp.sum(point), ))
                            )

        return X_new
    
    
    def generate(self, K, N, t1, t2, a0, b0, du=0.1, ratio=0.005, tol=0.01, seed=123):
        assert t1 < t2 < self.T
        
        key = random.PRNGKey(seed)

        X_list, Z_list = [], []
        pi_list, eta_list = [], []
        
        # t = 1
        success = False
        while not success:
            X, Z, pi_former, eta_former = self.__generate_initial(K, N, a0, b0, seed)
            
            Z_ind = jnp.argmax(Z, axis=1)
            indices, counts = jnp.unique(Z_ind, return_counts=True)
            
            if len(counts) != K:
                seed += 100
                continue
            
            #if jnp.min(counts) < 0.1 * N:
            #    seed += 100
            #    continue
            
            success = True
        
        X_list.append(X)
        Z_list.append(Z)
        
        # t = 2 - (t1-1)
        for t in range(2, t1): 
            X, Z = self.__generate_data_sequential_randomtrans(
                X, Z, eta=eta_former, ratio=ratio, seed=seed)
            X_list.append(X)
            Z_list.append(Z)
            
        # t = t1
        # change of eta
        u = dist.Uniform(-du, du).sample(key, sample_shape=(K, K))
        #u = dist.Uniform(-0.2, 0.2).sample(key, sample_shape=(K, K))
        eta_latter = eta_former + u
        eta_latter = eta_latter.at[eta_latter > 1.0].set(1.0 - tol)
        eta_latter = eta_latter.at[eta_latter < 0.0].set(tol)
        X = self.__generate_data_abrupt_eta(
                X, Z, eta_latter, seed=seed)
    
        X_list.append(X)
        Z_list.append(Z)
        
        # t = (t1+1) - (t2-1)
        #for t in range(t1+1, t2):
        # t = (t1+1) - t2
        for t in range(t1+1, t2+1):
            X, Z = self.__generate_data_sequential_randomtrans(
                       X, Z, eta=eta_latter, ratio=ratio, seed=seed)
            X_list.append(X)
            Z_list.append(Z)
            
        """
        # t = t2 
        # change of pi
        n_z = jnp.sum(Z, axis=0)
            
        idx_max_first, idx_max_second = n_z.argsort()[::-1][:2]
        n_z_max_first, n_z_max_second = jnp.sort(n_z)[::-1][:2]
        n_trans = int((n_z_max_first - n_z_max_second)/3)
    
        X, Z = self.__generate_data_abrupt_pi(
            X, Z, eta=eta_latter, n_trans=n_trans, 
            idx_before=idx_max_first, idx_after=idx_max_second, seed=seed)            
        
        X_list.append(X)
        Z_list.append(Z)
        """

        # t = (t2+1) - T 
        for t in range(t2+1, self.T+1):
            X, Z = self.__generate_data_sequential_randomtrans(
                X, Z, eta=eta_latter, ratio=ratio, seed=seed)

            X_list.append(X)
            Z_list.append(Z)
            
        return X_list, Z_list, eta_former, eta_latter
        

class DataGeneratorBSC:
    def __init__(self, T):
        """Data Generator for BSC
        Args:
        T (int): the time length for generated data
        """
        self.T = T
        
    def __generate_one(self, K, N, a, b, seed):
        # Z
        alpha = jnp.ones(K)
        pi = dist.Dirichlet(alpha).sample(random.PRNGKey(seed))
        Z = dist.Multinomial(probs=pi).sample(random.PRNGKey(seed), sample_shape=(N,))

        # Y
        rho = dist.Beta(a, b).sample(random.PRNGKey(seed))
        Y = dist.Bernoulli(rho).sample(random.PRNGKey(seed), sample_shape=(K, K))

        # X
        eta = dist.Beta(a, b).sample(random.PRNGKey(seed), sample_shape=(K, K))

        p1 = Z @ eta @ Z.T
        p2 = (Z @ (pi.reshape(-1, 1)))  @ (pi.reshape(1, -1) @ Z.T)
        connection_edge = Z @ Y.astype(int) @ Z.T

        X = numpyro.sample(
            'X',
            dist.Bernoulli(probs=connection_edge*p1+(1.0-connection_edge)*p2), 
            rng_key=random.PRNGKey(seed)
        )
        
        return X, Y, Z, pi, rho, eta       
        
    def generate(self, K, N, t1, t2, a0, b0, a1, b1, seed):
        X_list, Y_list, Z_list = [], [], []
        pi_list, rho_list, eta_list = [], [], []
        
        X, Y, Z, pi, rho, eta = generate_one(K, N, a0, b0, seed)
        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        pi_list.append(pi)
        rho_list.append(rho)
        eta_list.append(eta)
        
        for t in range(self.T, t1): 
            X_list.append(X)
            Y_list.append(Y)
            Z_list.append(Z)
            pi_list.append(pi)
            rho_list.append(rho)
            eta_list.append(eta)
            
        