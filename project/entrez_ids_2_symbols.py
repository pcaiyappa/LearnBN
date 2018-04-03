import mygene
import pandas as pd

def convert_entrez_to_symbol(entrezIDs):
    mg = mygene.MyGeneInfo()
    result = mg.getgenes(entrezIDs, fields='symbol', species='mouse')
    gene_symbols = [d['symbol'].lower() for d in result]
    return pd.Series(data=gene_symbols, index=entrezIDs)
