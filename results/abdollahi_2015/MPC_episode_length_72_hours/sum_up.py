import pandas as pd

data = list(pd.read_csv("abdollahi_12_.csv")["Supply_inlet_violation"][10:])

s = sum(data)/(72-10)

print(s)