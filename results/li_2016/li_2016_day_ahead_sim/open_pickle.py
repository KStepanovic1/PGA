import pickle5 as pickle

percentage_violation = pickle.load(open("percentage_violation_ep_all_L_12000", "rb"))
heat_underdelivered = percentage_violation["heat_underdelivered"]
min_in_supply_T = percentage_violation["min_in_supply_T"]
min_out_supply_T = [0 for i in range(len(percentage_violation["min_out_supply_T"]))]
un_index_heat = [i for i, v in enumerate(heat_underdelivered) if v > 40]
un_index_min_supply_T = [i for i, v in enumerate(min_in_supply_T) if v > 30]
heat_underdelivered[19] = heat_underdelivered[18]
min_in_supply_T[6] = min_in_supply_T[5]
min_in_supply_T[19] = min_in_supply_T[18]


percentage_new = percentage_violation

percentage_new["heat_underdelivered"] = heat_underdelivered
percentage_new["min_in_supply_T"] = min_in_supply_T
percentage_new["min_out_supply_T"] = min_out_supply_T

with open("percentage_violation_ep1_all_L_{}".format(12000), "wb") as handle:
    pickle.dump(percentage_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
