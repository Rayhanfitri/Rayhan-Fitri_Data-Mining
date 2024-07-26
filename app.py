import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Data transaksi penjualan tiket pesawat
Transactions = [
    ['Sriwijaya', 'Garuda', 'Lion Air'],
    ['Garuda', 'Lion Air', 'Sriwijaya'],  
    ['Lion Air', 'Sriwijaya', 'Garuda'],
    ['Sriwijaya', 'Lion Air', 'Garuda'],
    ['Batavia Air', 'Lion Air', 'Garuda', 'Sriwijaya', 'Garuda'],
    ['Lion Air', 'Sriwijaya', 'Batavia Air', 'Qatar Airways'],
    ['Lion Air', 'Sriwijaya', 'Garuda', 'Batavia Air'],
    ['Lion Air', 'Sriwijaya', 'Garuda', 'Qatar Airways'],
    ['Garuda', 'Lion Air', 'Sriwijaya'],
    ['Lion Air', 'Batavia Air', 'Garuda', 'Sriwijaya', 'Qatar Airways'],
    ['Lion Air', 'Garuda', 'Sriwijaya'],
    ['Garuda', 'Batavia Air', 'Lion Air', 'Sriwijaya'],
    ['Citilink', 'Sriwijaya', 'Batik Air', 'Lion Air'],
    ['Airasia'],
    ['Garuda', 'Sriwijaya'],
    ['Garuda', 'Lion Air', 'Sriwijaya', 'Citilink'],
    ['Citilink', 'Sriwijaya', 'Batik Air', 'Garuda', 'Lion Air'],
    ['Garuda', 'Sriwijaya', 'Citilink', 'Lion Air', 'Batik Air'],
    ['Lion Air', 'Sriwijaya', 'Citilink', 'Garuda', 'Batik Air'],
    ['Sriwijaya', 'Citilink', 'Lion Air', 'Batik Air', 'Garuda'],
    ['Batik Air', 'Sriwijaya', 'Citilink', 'Airasia', 'Garuda'],
    ['Lion Air', 'Garuda', 'Batik Air', 'Citilink'],
    ['Garuda', 'Sriwijaya', 'Lion Air'],
    ['Garuda', 'Sriwijaya', 'Lion Air', 'Batik Air'],
    ['Citilink', 'Batik Air', 'Sriwijaya', 'Garuda'],
    ['Batik Air'],
    ['Batik Air', 'Garuda', 'Lion Air', 'Sriwijaya'],
    ['Garuda', 'Lion Air', 'Batik Air', 'Citilink', 'Sriwijaya'],
    ['Lion Air', 'Sriwijaya Air', 'Garuda', 'Citilink', 'Batik Air'],
    ['Batik Air', 'Garuda', 'Lion Air'],
    ['Garuda', 'Batik Air'],
    ['Lion Air', 'Garuda', 'Citilink', 'Sriwijaya', 'Batik Air'],
    ['Lion Air', 'Garuda', 'Citilink', 'Batik Air'],
    ['Garuda', 'Citilink', 'Lion Air', 'Batik Air', 'Sriwijaya'],
    ['Sriwijaya', 'Citilink', 'Batik Air'],
    ['Citilink', 'Lion Air', 'Garuda', 'Batik Air']  
]

# Menggunakan TransactionEncoder untuk mengubah data menjadi format yang sesuai
te = TransactionEncoder()
te_ary = te.fit(Transactions).transform(Transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Menjalankan algoritma FP-Growth
frequent_itemsets = fpgrowth(df, min_support=0.3, use_colnames=True)

# Menghitung association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Streamlit app
st.title('Market Basket Analysis for Airline Tickets')

st.write('### Frequent Itemsets')
st.dataframe(frequent_itemsets)

st.write('### Association Rules')
st.dataframe(rules)
