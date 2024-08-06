import pandas as pd
data = pd.read_csv("data.csv")
data["arrTime"] = data["depTime"] + data["time"]
data.to_csv("ori_data.csv", index=False)
