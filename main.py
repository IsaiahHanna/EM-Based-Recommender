import numpy as np
import kmeans
import common
import naive_em
import em
import os

X = np.loadtxt(os.path.join(os.getcwd(),"Project 4","netflix","toy_data.txt"))
X_gold = np.loadtxt(os.path.join(os.getcwd(),"Project 4","netflix",'netflix_complete.txt'))


def run_kmeans():
    best_kmeans = []
    best_cost = []
    for k in [1,2,3,4]:
        cost = None
        for seed in [0,1,2,3,4]:
            mixture,post = common.init(X,k,seed)
            mixture,post,new_cost = kmeans.run(X,mixture,post)
            if seed == 0:
                cost = new_cost
                best_kmeans.append((mixture,post))
                best_cost.append(cost)
            elif cost > new_cost:
                cost = new_cost
                best_kmeans[k-1] = (mixture,post)
                best_cost[k-1] = cost
            else: 
                continue
        best_mix,best_post=best_kmeans[k-1]
        common.plot(X,best_mix,best_post,f"{k}")
        print(f"Best cost for {k}: {best_cost[k-1]}")


def run_em(naive:bool = False):
    best_em = []
    best_likelihoods = []
    
    for k in [1,2,3,4,5,6,7,8,9,10,11,12]:
        best_log_likelihood = None
        
        for seed in [0, 1, 2, 3, 4]:
            # Initialize mixture and posterior
            mixture, post = common.init(X, k, seed)

            # Run EM
            if naive:
                mixture, post, log_likelihood = naive_em.run(X, mixture, post)
            else:
                mixture, post, log_likelihood = em.run(X, mixture, post)

            # Track best run
            if seed == 0 or log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                if len(best_em) < k:
                    best_em.append((mixture, post))
                    best_likelihoods.append(log_likelihood)
                else:
                    best_em[k - 1] = (mixture, post)
                    best_likelihoods[k - 1] = log_likelihood

        # Plot and print result
        best_mix, best_post = best_em[k-1]
        #common.plot(X, best_mix, best_post, title=f"EM K={k}")
        print(f"\nBest log-likelihood for K={k}: {best_likelihoods[k-1]:.4f}")
        bic_score = common.bic(X,best_mix,best_likelihoods[k-1])
        print(f"BIC Score for K={k}: {bic_score}")

        if k == 12:
            X_pred = em.fill_matrix(X,best_mix)
            rmse = common.rmse(X_gold, X_pred)
            print(f"RMSE: {rmse}")




run_em(naive = False)
#run_kmeans()