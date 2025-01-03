import wn
import pandas as pa


# UMAP is a dimensionality reduction method like T-SNE but it scales up to larger data sets
from umap import UMAP

"""
This module provides prototype functions for dumping word embeddings and wordnets to file compatible
with large network analysis and visualisation software like networkX and cosmograph
"""

def kv2dataframe(kv):
    """
    Args:
        kv (KeyedVectors): a word embedding object from the gensim library

    Returns:
        a dataframe
    """
    #WARNING : this function is likely to be compute intensive !

    embeddings   = kv[kv.index_to_key]
    embeddings2D = UMAP(n_components=2).fit_transform(embeddings)
    return pa.DataFrame({'label'=kv.index_to_key,'x'=embeddings2D[:,0],'y'=embeddings2D[:,1]})

def wn2dataframe(wordnet):
    """
    Args:
        wordnet (wn.Wordnet): a Wordnet instance
    Returns:
        a couple of dataframes (edgelistframe,nodelistframe)
    """

    nodes = [ (synset.id,','.join(synset.lemmas())) for synset in wordnet.synsets() ]
    edges = []
    for src in wordnet.synsets():
        for relname,tgtlist  in src.relations().items():
            for tgt in tgtlist:
                edges.append( (src.id,tgt.id,relname) )

    return (pa.DataFrame.from_records(edges,columns=('source','target','rlabel')),
            pa.DataFrame.from_records(nodes,columns=('id','lemmas')))




if __name__ == '__main__':
    import gensim.downloader

    print(list(gensim.downloader.info()['models'].keys()))
    df = kv2dataframe(gensim.downloader.load('glove-wiki-gigaword-100'))
    df.to_csv('glove.tsv',sep='\t')


    wn_id             = 'oewn:2024'
    wordnet           = wn.Wordnet(wn_id)
    edges_df,nodes_df = wn2dataframe(wordnet)
    edges_df.to_csv(f'{wn_id}_edges.tsv',sep='\t')
    nodes_df.to_csv(f'{wn_id}_nodes.tsv',sep='\t')
