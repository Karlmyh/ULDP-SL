'''
The heavy hitter class for identifying important variables.
'''

import numpy as np
import math
from scipy.linalg import hadamard


def get_hash(v, logd, hash_list):
    '''
    Get the hash value of v.
    '''
    if len(v) == logd:
        intv = int(v, 2)
    elif len(v) < logd:
        addition = sum([2 ** (logd - i ) for i in range(logd - len(v)) ])
        intv = int(v, 2) + addition
    elif len(v) == 0:
        intv = sum([2 ** (logd - i ) for i in range(logd - len(v)) ])
    return hash_list[intv]


def random_response(x, epsilon):
    '''
    Randomly flip the sign of x with probability exp(epsilon) / (exp(epsilon) + 1).
    '''
    prob = 1 / (math.exp(- epsilon) + 1)
    flag = np.random.choice([True, False], p = [prob, 1 - prob])
    if flag:
        return x
    else:
        return -x
    
def intersect(a, b):
    '''
    Take the intersection of two lists.
    '''
    if len(a) == 0 or len(b) == 0:
        return []
    else:
        return list(set(a) & set(b))


def ChildSet(Prefixes):
    '''
    Input:
    ---------
    Prefixes, the set of prefixes

    Output:
    ---------
    ChildSet, the set of child prefixes
    '''
    ChildSet = []
    if len(Prefixes) == 0:
        return ["0", "1"]
    else:
        for v in Prefixes:
            ChildSet.append(v + "0")
            ChildSet.append(v + "1")
        return ChildSet





class HeavyHitters(object):
    '''
    The heavy hitters class.
    By Raef Bassily et al., Practical locally private heavy hitters.
    '''

    def __init__(self, 
                 epsilon, 
                 d, 
                 user_values, 
                 min_hitters = 2,
                 alpha = 0.1, 
                 t = None, 
                 m = None, 
                 beta = 0.05, 
                 if_est_freq = False, 
                 random_state = None
                 ):
        '''
        Initialize the heavy hitters class.

        Parameters
        ----------
        d : int
            The number of possible values.
        user_values : length n list.
            The list of user values. Containing binary strings with length less than logd.
        t : int
            The number of random hashes.
        m : int
            Size of the Hadamard matrix.
        epsilon : float
            The privacy budget.
        beta : float
            Failure probability.
        if_est_freq : bool
            If estimate the frequency of the heavy hitters. Will waste half privacy budgets.
        random_state : int
            The random state.
        min_hitters : int
            The minimum number of heavy hitters.


        Attributes
        ----------
        d : int
            The number of possible values.
        logd : int
            The log of d.
        user_values : length n list.
            The list of user values. Containing binary strings with length less than logd.
        t : int
            The number of random hashes.
        m : int
            Size of the Hadamard matrix.
        epsilon : float
            The privacy budget.
        beta : float
            Failure probability.
        n : int
            The number of users.
        l_vec : length n list.
            Index of bit taking values in [0, log d].
        j_vec : length n list.
            Index of hash function taking values in [0, t].
        r_vec : length n list.
            Index of Hadamard matrix taking values in [0, m].
        h_vec : length t list.
            The list of h (n to m) hash functions.
        g_vec : length t list.
            The list of g (n to +1-1) hash functions.
        '''
        # constants
        self.d = d
        self.logd = int(np.ceil(math.log(d, 2)) )
        self.user_values = [f'{integers:0{self.logd}b}' for integers in user_values]
        self.epsilon = epsilon
        self.alpha = alpha
        self.if_est_freq = if_est_freq
        self.min_hitters = min_hitters
        if self.if_est_freq:
            self.aepsilon = (math.exp(- self.epsilon / 2) + 1) / (1 - math.exp(- self.epsilon / 2))
        else:
            self.aepsilon = (math.exp(- self.epsilon ) + 1) / (1 - math.exp(- self.epsilon ))
        self.beta = beta
        self.random_state = random_state
        self.n = len(user_values)
        if t is None:
            # self.t = int(110 * math.log(self.n / self.beta, 2))
            self.t = int(math.log(self.n / self.beta, 2))
        if m is None:
            # self.m = int(48 * math.sqrt(self.n / math.log(self.n / self.beta, 2)))
            self.m = int(math.sqrt(self.n / math.log(self.n / self.beta, 2)))
            self.m = int(2**int(math.log(self.m, 2))) 
        # self.eta = 147 * math.sqrt(self.n * math.log(self.n / self.beta, 2) * math.log(self.d, 2))
        # abandoned
        self.eta = math.sqrt(self.n * math.log(self.n / self.beta, 2) * math.log(self.d, 2)) /8

        flag = 1
        while flag:
            self.l_vec = np.random.choice(self.logd, self.n)
            counts = np.unique(self.l_vec, return_counts = True)[1]
            if len(counts) == self.logd and counts.min() > 2:
                flag = 0
        
        flag = 1 
        while flag:
            self.j_vec = np.random.choice(self.t, self.n)
            counts = np.unique(self.j_vec, return_counts = True)[1]
            if len(counts) == self.t and counts.min() > 2:
                flag = 0
        self.r_vec = np.random.choice(self.m, self.n)

        self.h_vec = [np.random.choice(self.m, int(2**(self.logd + 1)), replace = True) for _ in range(self.t)]
        self.g_vec = [np.random.choice([-1, 1], int(2**(self.logd + 1)), replace = True) for _ in range(self.t)]

        self.W = hadamard(self.m) 
        
    
    def LocalRnd(self, i, final):
        '''
        Input: 
        ---------
        i, index of user
        final, the flag

        Output:
        ---------
        y, the local randomizer output
        '''
        v = self.user_values[i]
        l, j, r = self.l_vec[i], self.j_vec[i], self.r_vec[i]
        if not final:
            s = get_hash(v[:(l + 1)], self.logd, self.g_vec[j])
            c = get_hash(v[:(l + 1)], self.logd, self.h_vec[j])
        else:
            s = get_hash(v, self.logd, self.g_vec[j])
            c = get_hash(v, self.logd, self.h_vec[j])
        x = s * self.W[r, c]

        if self.if_est_freq:
            return random_response(x, self.epsilon / 2)
        else:
            return random_response(x, self.epsilon)
    

    def FreqOracle(self, l, V_hat, I, gamma, final):    
        '''
        Input:
        ---------
        l, the index of bit
        V_hat, the set of prefixes
        I, the index of samples, a dictionary, key is the js in 1 to t, value is the associated index of sample
        gamma, scaling factor
        final, the flag

        Output:
        ---------
        Freqlist, the list of frequency pairs
        '''
        Freqlist = {}
        for v in V_hat:
            assert len(v) == l + 1, "The length of v is {}, namely {}, but l is {}. Moreover, v dic is {}".format(len(v), v, l, V_hat)
            fv_list = []
            for j in range(self.t):
                s = get_hash(v, self.logd, self.g_vec[j])
                c = get_hash(v, self.logd, self.h_vec[j])
                fjv = 0
                if len(I[j]) != 0:
                    for i in I[j]:
                        y = self.LocalRnd(i, final)
                        r = self.r_vec[i]
                        fjv += y * s * self.W[r, c]
                    fjv *= gamma * self.aepsilon
                    fv_list.append(fjv)
            fv = np.median(fv_list)
            Freqlist[v] = fv
        return Freqlist

    def TreeHist(self):
        '''
        V, the user values, a list. 
        '''
        Ij = { n:rep[n] for rep in [{}] for i,n in enumerate(self.j_vec)  if rep.setdefault(n,[]).append(i) or len(rep[n])==2 }
        Il = { n:rep[n] for rep in [{}] for i,n in enumerate(self.l_vec)  if rep.setdefault(n,[]).append(i) or len(rep[n])==2 }
        gamma = self.t * self.logd
        Prefixes = []

        for l in range(self.logd):
            ChildSetPrefixes = ChildSet(Prefixes)
            try:
                Ijl = [intersect(Ij[j], Il[l]) for j in range(self.t)]
            except:
                raise ValueError("Ij: {}, Il: {}, l: {}, lvec : {}".format(Ij, Il, l, self.l_vec))
            Freqlist_l = self.FreqOracle(l, ChildSetPrefixes, Ijl, gamma, False)
            num_keep_min = np.ceil(max(self.min_hitters / 2**(self.logd - l - 1), 1)).astype(int)
            num_keep_min = min(num_keep_min, len(Freqlist_l))
            threshold_max = np.quantile(list(Freqlist_l.values()), 1 - num_keep_min / len(Freqlist_l))
            num_keep_max = min(self.min_hitters, len(Freqlist_l))
            threshold_min = np.quantile(list(Freqlist_l.values()), 1 - num_keep_max / len(Freqlist_l))

            threshold = min(np.floor(threshold_max), self.n * self.alpha)
            threshold = max(threshold, np.floor(threshold_min))
            
            NewPrefixes = [v for v in ChildSetPrefixes if Freqlist_l[v] >= threshold]
            assert NewPrefixes != [], "NewPrefixes is empty, threshold: {}, v : {}".format(threshold, Freqlist_l) 
            Prefixes = NewPrefixes
 
        # if estimate frequency, return a dict, else return the list of heavy hitters
        if self.if_est_freq:
            gamma = self.t
            SuccHist = self.FreqOracle(self.logd - 1, Prefixes, Ij, gamma, True)
            return SuccHist
        else:
            return Prefixes


    def apply(self):
        '''
        Apply the heavy hitters algorithm.
        '''
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.if_est_freq:
            SuccHist = self.TreeHist()
            SuccHist = {k: v for k, v in SuccHist.items() if int(k, 2) < self.d}
            heavy_hitters = [int(k, 2) for k,_ in SuccHist.items()]
            fequency = [v for _,v in SuccHist.items()]
            return heavy_hitters, fequency
        else:
            Prefixes = self.TreeHist()
            heavy_hitters = [int(k, 2) for k in Prefixes]
            heavy_hitters = [hitter for hitter in heavy_hitters if hitter < self.d]
            if len(heavy_hitters) == 0:
                heavy_hitters = [0]
            return heavy_hitters, None

        

