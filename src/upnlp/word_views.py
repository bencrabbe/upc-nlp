import wn
import pandas as pa


# UMAP is a dimensionality reduction method like T-SNE but it scales up to larger data sets
from umap import UMAP


__all__ = ['emb2df','wn2df']

"""
This module provides prototype functions for dumping word embeddings and wordnets to file compatible
with large network analysis and visualisation software like networkX and cosmograph
"""

def emb2df(kv,filter_list=None):
    """
    This function projects n-dimensional embeddings to 2D and returns a dataframe mapping word strings to 2d vectors

    Args:
        kv (KeyedVectors) : a word embedding object from the gensim library
        filter_list (list): list of strings, if not None, the output will have a vocabulary that results from
                             the intersection with this list.
    Returns:
        a dataframe
    """
    if filter_list:
        filter_list = set(filter_list)
        vocabulary = [wf for wf in kv.index_to_key if wf in filter_list]

    embeddings   = kv[vocabulary]
    embeddings2D = UMAP(n_components=2).fit_transform(embeddings)
    return pa.DataFrame({'id':vocabulary,'x':embeddings2D[:,0],'y':embeddings2D[:,1]})

def wn2df(wordnet):
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

    wn_id             = 'oewn:2024'
    wordnet = wn.Wordnet(wn_id)

    df = emb2df(gensim.downloader.load('glove-wiki-gigaword-100'),filter_list= [word.lemma() for word in wordnet.words()])
    df.to_csv('glove.tsv',sep='\t')

    edges_df,nodes_df = wn2df(wordnet)
    edges_df.to_csv(f'{wn_id}_edges.tsv',sep='\t')
    nodes_df.to_csv(f'{wn_id}_nodes.tsv',sep='\t')
