import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

with open("files/Transacoes.txt") as f:
    transactions = [line.strip().split(",") for line in f.readlines()]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print(df)

frequent_itemset = apriori(df, min_support = 0.5, use_colnames=True)
print(frequent_itemset)

rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.5)
print("rules: ", rules)