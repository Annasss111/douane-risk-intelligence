import pandas as pd
tr = pd.read_csv("df_syn_train_eng.csv")
va = pd.read_csv("df_syn_valid_eng.csv")
te = pd.read_csv("df_syn_test_eng.csv")
print(tr.shape, va.shape, te.shape)
print(tr.columns.tolist() == va.columns.tolist())
# check if there's a shared ID that could tie back to one original file
print(tr['Declaration ID'].is_unique, te['Declaration ID'].is_unique)
