import pandas as pd
import numpy as np


# set_a = pd.read_csv('data/csvs/set_a.csv')
# names = set_a['fname'].values
# names = [x.split('/')[-1] for x in names]
# labels = set_a['label']

# df_A = pd.DataFrame()
# df_A['name'] = names
# df_A['label'] = labels 
# df_A.to_csv('data/csvs/A.csv',index=False)

# -------------------------------------------------
# set_b = pd.read_csv('data/csvs/set_b.csv')
# print set_b.head()
# print set_b.shape

# names = set_b['fname'].values
# names = [x.split('/')[-1] for x in names]
# labels = set_b['label']

# df_B = pd.DataFrame()
# df_B['name'] = names
# df_B['label'] = labels 
# df_B.to_csv('data/csvs/B.csv',index=False)
# -------------------------------------------------

# df_A = pd.read_csv('data/csvs/A.csv')
# df_B = pd.read_csv('data/csvs/B.csv')

# df = pd.concat([df_A,df_B])
# df.to_csv('data/csvs/all.csv',index=False)

# -------------------------------------------------

# df = pd.read_csv('data/csvs/all.csv')
# labels = df['label']

# items,counts = np.unique(labels,return_counts=True)

# for item,count in zip(items,counts):
# 	print item,':',count

# -------------------------------------------------

