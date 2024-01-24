try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np

ep5 = pickle.load(open("percentage_violation_ep_all_L_12000", "rb"))
min_in_supply_T = ep5["min_in_supply_T"]
print(min_in_supply_T)
# for i in range(len(min_in_supply_T)):
#    if np.isnan(min_in_supply_T[i]):
#        print(i)
