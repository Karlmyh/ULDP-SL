

from ULDPFS import HeavyHitters
import numpy as np

def test_heavy_hitters():
    epsilon = 4
    d = 10000
    user_values = np.random.choice(10, 300)
    heavy_hitters = HeavyHitters(epsilon, d, user_values, min_hitters= 10)
    hitter_dic = heavy_hitters.apply()
    print(hitter_dic)

    heavy_hitters = HeavyHitters(epsilon, d, user_values, if_est_freq = True, min_hitters= 10)
    hitter_dic = heavy_hitters.apply()
    print(hitter_dic)


    



    
