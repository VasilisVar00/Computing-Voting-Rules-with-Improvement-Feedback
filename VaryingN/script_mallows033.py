import numpy as np
import itertools
import random
import csv
from math import sqrt,exp
from pref_voting.profiles import Profile
from pref_voting.generate_profiles import generate_profile




# ---- definition of different probability distributions for feedback ----
# take the POSITION j of a candidate and hop parameter t
# return list of POSITIONS of reachable candidates and probabiity vector
def lin_prob(j,t):
    if j == 0:
        return [0], [1.0]
    else:
        l = np.array([i+1 for i in range(min(j,t))])
        norm = sum(l)
        probs = l / norm
        return range(j-min(j,t),j), probs

def exp_prob(j,t):
    if j == 0:
        return [0], [1.0]
    else:
        l = np.array([exp(i+1) for i in range(min(j,t))])
        norm = sum(l)
        probs = l / norm
        return range(j-min(j,t),j), probs

def unif_prob(j,t):
    if j == 0:
        return [0], [1.0]
    else:
        l = np.ones(min(j,t))
        norm = sum(l)
        probs = l / norm
        return range(j-min(j,t),j), probs

def get_probs(j,t, type="unif"):
    if type == "unif":
        return unif_prob(j,t)
    elif type == "exp":
        return exp_prob(j,t)
    elif type == "lin":
        return lin_prob(j,t)

# ---- feedback and score calculation ----
def calculate_feedback(rankings, t=1, mode="improvement", type="unif"):
    M = len(rankings[0])
    d = [[0 for _ in range(M)] for _ in range(M)] # d[a][b] stores # of a above b
    plu = {i:0 for i in range(M)}
    if mode == "improvement":
        for r in rankings:
            candidate = random.choice(r) # choose a candidate from ranking at random
            j = r.index(candidate) # get its position in the list
            pos,probs = get_probs(j,t,type=type)
            i = np.random.choice(pos,size=(1,1),p=probs)[0][0]
            a,b = r[i], r[j]
            if a == b:
                plu[a] += 1
                continue

            d[a][b] += 1
        for i in plu.keys():
            for b in range(M):
                if i == b:
                    continue
                else:
                    d[i][b] += plu[i]
                

    if mode == "pairwise":
        for r in rankings:
            c1 = random.choice(r)
            c2 = random.choice(r)
            # ensures that c1, c2 are different
            while(c2 == c1):
                c2 = random.choice(r)
            if(r.index(c1) > r.index(c2)):
                a,b = (c2,c1)
            else:
                a,b = (c1, c2)
            d[a][b] += 1
    return plu,d

# returns winner and its score under scoring rule
def find_true_scores(rankings, rule):
    prof = Profile(rankings)

    if(rule == "borda" or rule == "BORDA"):
        return prof.borda_scores()
    elif(rule == "copeland" or rule == "COPELAND"):
        return prof.copeland_scores(scores=(1,1/2,0))
    elif(rule == "plurality" or rule == "PLURALITY"):
        return prof.plurality_scores()

    

# returns {candidate : score} dictionary
# in a pairwise ballot the convention is that (i,j) means i above j
def find_pairwise_scores(d,M,rule="borda"):
    if rule == "copeland" or rule == "COPELAND":
        copl = {i:0 for i in range(M)}

        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                else:
                    if d[i][j] == 0 and d[j][i] == 0:
                        continue
                    elif d[i][j] == d[j][i]:
                        copl[i] += 1/2 * ((d[i][j]) / (d[i][j] + d[j][i]))
                    elif d[i][j] > d[j][i]:
                        copl[i] += 1 * ((d[i][j]) / (d[i][j] + d[j][i]))
        return copl
    
    elif(rule == "borda" or rule == "BORDA"):
        borda_d = {i:0 for i in range(M)}
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                else:
                    if d[i][j] == 0 and d[j][i] == 0:
                        continue
                    borda_d[i] += d[i][j] / (d[i][j] + d[j][i])
        return borda_d
    


        



# ---- parameters of the experiment ----

M = 20 # number of candidates
phi = 1/3 # parameters of mallows model
t = 5 # hop parameter in improvement feedback
it = 500 # number of iterations.
ref_perm = range(M) # reference permutation for Mallows model
copeland_vector = (1,1/2,0) # Copeland "coefficients" for win, draw, defeat
rules = ["borda", "copeland", "plurality"]  
modes =["improvement", "pairwise"]
types = ["unif", "lin", "exp"]
N = range(50, 1050, 50)




l = list(itertools.product(modes,rules,types))
print(f"Current value of t is {t}...")
print(f"Current value of phi is {phi}...")
means = {(i,j,k):[] for (i,j,k) in l}
lcbs = {(i,j,k):[] for (i,j,k) in l}
ucbs= {(i,j,k):[] for (i,j,k) in l}
for n in N:
    res = {(i,j,k):[] for (i,j,k) in l}
    for _ in range(it):
        prof = generate_profile(M, n,
                        probmodel="mallows",
                        phi=phi, central_vote=ref_perm)
        rankings = prof.rankings
        for rule in rules:
            ts = find_true_scores(rankings,rule)
            true_key = max(ts, key = ts.get)
            for mode in modes:
                if mode == "improvement":
                    for type in types:
                        plu, d = calculate_feedback(rankings,t=t,mode=mode,type=type)
                        if rule == "plurality":
                            estimate_key = max(plu, key = plu.get)
                            res[(mode,rule,type)].append(ts[estimate_key] / ts[true_key])
                        else:
                            scores = find_pairwise_scores(d,M,rule)
                            estimate_key = max(scores, key = scores.get)
                            res[(mode,rule,type)].append(ts[estimate_key] / ts[true_key])

                if mode == "pairwise":
                    if rule == "plurality":
                        continue
                    else:
                        plu, d = calculate_feedback(rankings,t=t,mode=mode)
                        scores = find_pairwise_scores(d,M,rule)
                        estimate_key = max(scores, key = scores.get)
                        res[(mode,rule,"unif")].append(ts[estimate_key] / ts[true_key])



                            


    for rule in rules:
        for mode in modes:
            if mode == "pairwise":
                mean = np.average(res[(mode,rule,"unif")])
                std = np.std(res[(mode, rule,"unif")])
                means[(mode,rule,"unif")].append(mean)
                ucbs[(mode,rule,"unif")].append(mean + (1.96 * (std / sqrt(it))))
                lcbs[(mode,rule,"unif")].append(mean - (1.96 * (std / sqrt(it))))
            else:
                for type in types:
                    mean = np.average(res[(mode,rule,type)])
                    std = np.std(res[(mode, rule,type)])
                    means[(mode,rule,type)].append(mean)
                    ucbs[(mode,rule,type)].append(mean + (1.96 * (std / sqrt(it))))
                    lcbs[(mode,rule,type)].append(mean - (1.96 * (std / sqrt(it))))



with open(f"results0.csv", "a", newline= '') as file:
    writer = csv.writer(file)
    field = ["t","phi","n","mode","rule","type","mean","ucb","lcb"]
    writer.writerow(field)
for n in N:
    nid = N.index(n)
    for rule in rules:
        for mode in modes:
            if mode == "pairwise":
                if rule == "plurality":
                    continue
                else:
                    with open("results0.csv", "a", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([t, phi,n,mode,rule,"unif",means[mode,rule,"unif"][nid],ucbs[mode,rule,"unif"][nid],lcbs[mode,rule,"unif"][nid] ] )
            else:
                for type in types:
                    with open("results0.csv", "a", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([t, phi,n,mode,rule,type,means[mode,rule,type][nid],ucbs[mode,rule,type][nid],lcbs[mode,rule,type][nid] ] )
                    






           


      
                


                    
                





