import numpy as np
import torch
import torch.nn.functional as F


class GaussianHMM:

    def __init__(self, n_states=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def init_params(self, n_features):
        torch.manual_seed(self.random_state)
        K, D = self.n_states, n_features

        # initial state distribution: uniform
        self.log_pi = torch.log(torch.ones(K) / K)

        # transition matrix: slight self-preference (regimes tend to persist)
        A = torch.ones(K, K) + 5.0 * torch.eye(K)
        self.log_A = torch.log(A / A.sum(dim=1, keepdim=True))

        # Gaussian parameters per state
        self.mu = torch.randn(K, D) * 0.01
        # start with identity covariances
        self.sigma = torch.stack([torch.eye(D) * 0.1 for _ in range(K)])

    def log_gaussian(self, X):
        T, D = X.shape
        K = self.n_states
        log_prob = torch.zeros(T, K)

        for k in range(K):
            dist = torch.distributions.MultivariateNormal(
                loc=self.mu[k],
                covariance_matrix=self.sigma[k]
            )
            log_prob[:, k] = dist.log_prob(X)

        return log_prob

    def forward_log(self, log_emit):
        T, K = log_emit.shape
        alpha_log = torch.zeros(T, K)

        
        alpha_log[0] = self.log_pi + log_emit[0]
        
        for t in range(1, T):
            prev = alpha_log[t - 1].unsqueeze(1) + self.log_A  
            alpha_log[t] = torch.logsumexp(prev, dim=0) + log_emit[t]

        log_likelihood = torch.logsumexp(alpha_log[-1], dim=0)
        return alpha_log, log_likelihood

    def backward_log(self, log_emit):
        T, K = log_emit.shape
        beta_log = torch.zeros(T, K)

        beta_log[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            next_ = self.log_A + log_emit[t + 1].unsqueeze(0) + beta_log[t + 1].unsqueeze(0)
            beta_log[t] = torch.logsumexp(next_, dim=1)

        return beta_log

    ##  Fit the HMM using EM
    def fit(self, X):
        X = torch.FloatTensor(X)
        T, D = X.shape
        self.init_params(D)

        prev_ll = -float('inf')

        for i in range(self.n_iter):
            #  Eexpectation step
            log_emit = self.log_gaussian(X)                        
            alpha_log, log_ll = self.forward_log(log_emit)         
            beta_log = self.backward_log(log_emit)                

            log_gamma = alpha_log + beta_log
            log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
            gamma = torch.exp(log_gamma)                            

            log_xi = torch.zeros(T - 1, self.n_states, self.n_states)
            for t in range(T - 1):
                log_xi[t] = (
                    alpha_log[t].unsqueeze(1) + self.log_A + log_emit[t + 1].unsqueeze(0) 
                    + beta_log[t + 1].unsqueeze(0)
                )
                log_xi[t] -= torch.logsumexp(log_xi[t].reshape(-1), dim=0)
            xi = torch.exp(log_xi)                                

           
            # update initial distribution
            self.log_pi = torch.log(gamma[0] + 1e-8)

            # update transition matrix
            A_new = xi.sum(dim=0) + 1e-8                         
            self.log_A = torch.log(A_new / A_new.sum(dim=1, keepdim=True))

            # update gausian parameters
            gamma_sum = gamma.sum(dim=0) + 1e-8                  
            self.mu = (gamma.unsqueeze(2) * X.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(1)

            for k in range(self.n_states):
                diff = X - self.mu[k]                              
                weighted = gamma[:, k].unsqueeze(1) * diff        
                cov = (weighted.T @ diff) / gamma_sum[k]
                self.sigma[k] = cov + 1e-4 * torch.eye(D)

            # check for convergence
            improvement = log_ll.item() - prev_ll
            print(f"  HMM iter {i+1:>3d} | log-likelihood: {log_ll.item():.4f} | improvement: {improvement:.6f}")
            if improvement < self.tol:
                print(f"  Converged at iteration {i+1}")
                break
            prev_ll = log_ll.item()

        return self

    # viterbi algo
    def predict(self, X):
        X = torch.FloatTensor(X)
        T, _ = X.shape
        K = self.n_states

        log_emit = self.log_gaussian(X)                      

        viterbi = torch.zeros(T, K)
        backptr = torch.zeros(T, K, dtype=torch.long)

        viterbi[0] = self.log_pi + log_emit[0]

        for t in range(1, T):
            trans = viterbi[t - 1].unsqueeze(1) + self.log_A     
            best_prev = trans.max(dim=0)
            viterbi[t] = best_prev.values + log_emit[t]
            backptr[t] = best_prev.indices

        
        states = torch.zeros(T, dtype=torch.long)
        states[T - 1] = viterbi[T - 1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        return states.numpy()

    # for distribution accross different regimes
    def get_gamma(self, X):
        X = torch.FloatTensor(X.copy())
        log_emit = self.log_gaussian(X)
        alpha_log, _ = self.forward_log(log_emit)
        beta_log = self.backward_log(log_emit)
        log_gamma = alpha_log + beta_log
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
        return torch.exp(log_gamma).numpy() 


def get_regime_weights(model, n_states=3):
    weights = torch.zeros(n_states)
    for k in range(n_states):
        mu = model.mu[k]
        mean_ret = mu[0].item()
        vol = mu[1].item()
        bear_score = -mean_ret + vol
        weights[k] = bear_score
    w_min, w_max = weights.min(), weights.max()
    weights = 0.5 + 1.5 * (weights - w_min) / (w_max - w_min + 1e-8)
    return weights


def prepare_regimes(df_train, df_val, df_test, n_states=3, n_iter=100, tol=1e-4, random_state=42):
    """
    Fit HMM on training data and label regimes for all splits.
    Adds a regime col to each split
    Observations: daily (cross-sectional mean return, cross-sectional vol)
    """
    def daily_features(df):
        return (
            df.groupby('date')['ret']
            .agg(['mean', 'std'])
            .rename(columns={'mean': 'mean_ret', 'std': 'vol'})
            .fillna(0)
            .sort_index()
        )

    train_daily = daily_features(df_train)
    X_train = train_daily.values
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1 
    X_train = (X_train - mean) / std


    model = GaussianHMM(n_states=n_states, n_iter=n_iter, tol=tol, random_state=random_state)
    model.fit(X_train)
    print("\nLearned state centers (normalized scale):")
    for k in range(model.n_states):
        mu = model.mu[k].numpy()
        print(f"  State {k}: mean_ret={mu[0]:.3f}, vol={mu[1]:.3f}")

    def label(df):
        daily = daily_features(df)
        X = (daily.values - mean) / std
        regime_seq = model.predict(X)
        gamma = model.get_gamma(X)  

        date_to_regime = dict(zip(daily.index, regime_seq))
        date_to_probs = {
            date: gamma[i] for i, date in enumerate(daily.index)
        }

        df = df.copy()
        df['regime'] = df['date'].map(date_to_regime).astype(int)
        df['regime_prob_0'] = df['date'].map(lambda d: date_to_probs[d][0])
        df['regime_prob_1'] = df['date'].map(lambda d: date_to_probs[d][1])
        df['regime_prob_2'] = df['date'].map(lambda d: date_to_probs[d][2])
        return df

    df_train = label(df_train)
    df_val   = label(df_val)
    df_test  = label(df_test)

    print(f"Regime distribution (train): {df_train.groupby('regime').size().to_dict()}")
    weights = get_regime_weights(model, n_states)
    print(f"Regime weights: {weights.tolist()}")
    return model, df_train, df_val, df_test, weights