import pandas as pd
import numpy as np
import ast

df = pd.read_csv('test_data1.csv')
item_id = df['item_seq_item_id'].apply(ast.literal_eval)
category_id = df['item_seq_category_id'].apply(ast.literal_eval)

data = pd.DataFrame({'item_id':item_id,
                     'category_id':category_id
                     })

data_expanded = data.apply(lambda x: x.apply(pd.Series).stack()).reset_index(drop=True)
data_expanded.columns = ['item_id', 'category_id']
data_unique = data_expanded.drop_duplicates()

sorted_df = data_unique.sort_values(by='item_id')
print(sorted_df)

sorted_df.to_csv('item_data1.csv',index=False)