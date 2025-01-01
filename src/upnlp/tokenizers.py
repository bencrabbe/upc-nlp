import torch
import json
import os
import hashlib
import re
import wn
import inflect
import nlp_datasets
import bpemb

__all__ = ['normalize_text','AutoTokenizer','WordNetTokenizer','PunctuationTokenizer']

__MORPH_GEN__ = inflect.engine()


def normalize_text(text):
    """
    For tokenizers splitting tokens on blank spaces it is relevant to normalize spacing before tokenization.
    Case is lowered too.
    Args:
        text (str): the text to normalize
    Returns:
         the normalized text
    """
    translation_map = str.maketrans({'.':' . ','?':' ? ','!':' ! ',',':' , ' })
    text = text.translate(translation_map)
    return " ".join(text.split()).lower()

def default_en_wordlist():
    #downloads a basic English wordlist of 10000 word forms
    hub  = nlp_datasets.HubUpc()
    path = os.path.join(hub.get_local_path('datasets/enwords'),'enwordlist.txt')
    with open(path) as infile:
        wordlist = infile.read().split()
    return wordlist




class AutoTokenizer:
    """
    This is a namespace for easy loading of tokenizers defined from this module.
    """

    @staticmethod
    def from_pretrained(dirpath):
        """
        Loads the tokenizer from the model directory

        Args:
          dirpath (path or string) : path to the tokenizer params dir

        Returns:
          a Tokenizer object
        """
        with open(os.path.join(dirpath, 'tokenizer.json')) as infile:
            ldict = json.loads(infile.read())
        if ldict['ClassName'] == 'PunctuationTokenizer':
                return PunctuationTokenizer.from_pretrained(dirpath)

        if ldict['ClassName'] == 'WordNetTokenizer':
                return WordNetTokenizer.from_pretrained(dirpath)

        if ldict['ClassName'] == 'BpeTokenizer':
                return BpeTokenizer.from_pretrained(dirpath)
        raise Exception("There is something wrong. I cannot get the type of the tokenizer")


class AbstractTokenizer:
    """
    This is the abstract superclass for the tokenizers in this module
    """
    def __init__(self,unk='<unk>',pad='<pad>',bos=None,eos=None):
        """
        Args:
            unk(str): string for unknown tokens
            pad(str): string for padding tokens
            eos(str): string for eos token
            bos(str): string for bos token
        """
        self.unk = unk
        self.pad = pad
        self._bos, self._eos = bos, eos
        self.vocabulary = []
        self.types2idx = {}


    def signature(self):
        """
        Returns an md5 checksum of the vocabulary in this segmenter.
        Can be used to test if two segmenters perform the exact same segmentation on the same input

        Returns:
            a string
        """
        vocab_hash = hashlib.md5()
        for word in self.vocabulary:
            vocab_hash.update(word.encode())
        return vocab_hash.hexdigest()


    def save_pretrained(self,dirpath):
        """
        Saves the tokenizer to model dir.

        Args:
          dirpath (path or string) : path to the tokenizer params dir
        """
        with open(os.path.join(dirpath, 'tokenizer.json'), 'w') as outfile:
            outfile.write(json.dumps({'unk': self.unk, 'pad': self.pad, 'vocabulary': self.vocabulary,
                                      'bos': self._bos,'eos': self._eos,'ClassName':self.__class__.__name__}))


    def add_tokens(self, tokens):
        """
        Adds a list of tokens to the vocabulary.

        Args:
          tokens :  a list of strings to add to the vocabulary
        """
        if not type(tokens) == list:
            raise Exception("Error tokens are not given as a list. Cannot continue anymore")

        self.vocabulary = list(sorted(set(self.vocabulary).union(set(tokens))))
        self.types2idx = {elt: idx for idx, elt in enumerate(self.vocabulary)}

    def tokenize(self, string):
        """
        Splits a string into tokens

        Args:
          string : a string to tokenize

        Returns:
          a list of strings
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """
        Maps a list of tokens to integer codes

        Args:
          tokens : a list of strings
        Returns:
          a list of integers
        """
        unkid = self.types2idx[self.unk]
        return [self.types2idx.get(token, unkid) for token in tokens]

    def encode(self, string):
        """
        Encodes a string into a list of integers

        Args:
          string : a text to encode
        Returns:
          a list of integers
        """
        tokens = self.tokenize(string)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids):
        """
        Decodes a list of integers into a string

        Args:
          ids : a list of integers
        Returns:
          a string
        """
        tokens = [self.vocabulary[idx] for idx in ids]
        return ' '.join(tokens)

    def __call__(self, string):
        """
        @see the encode method
        """
        return self.encode(string)

    @property
    def pad_id(self):
        """
        the id of the pad token
        """
        return self.types2idx[self.pad]

    @property
    def bos_token(self):
        """
        the string used for the begin of sentence token
        """
        if self._bos is None:
            raise Exception("Warning trying to use the bos token while it is undefined for the tokenizer")
        return self._bos

    @property
    def eos_token(self):
        """
        the string used for the end of sentence token
        """
        if self._eos is None:
            raise Exception("Warning trying to use the eos token while it is undefined for the tokenizer")
        return self._eos

    @property
    def eos_token_id(self):
        """
        the integer used for the end of sentence token
        """
        return self.types2idx[self.eos_token]

    @property
    def vocab_size(self):
        """
        the size of the vocabulary
        """
        return len(self.vocabulary)

    def pad_batch(self, batch_codes):
        """
        Pads a batch of integers with the pad code

        Args:
          batch_codes : a list of lists of integers

        Returns:
          a list of lists of integers as a tensor
        """
        max_len = max([len(sentence) for sentence in batch_codes])
        padded_codes = [sentence + [self.pad_id] * (max_len - len(sentence)) for sentence in batch_codes]
        return torch.LongTensor(padded_codes)


class BpeTokenizer (AbstractTokenizer):
    """
    This is a byte pair encoding (bpe) sentence piece tokenizer. It splits sentences into subword units.
    It wraps the bpemb package @see https://bpemb.h-its.org
    """
    def __init__(self,lang='en',vocabulary_size=10000,emb_size=300,pad='<pad>',bos=None,eos=None):
        """
        Creates a tokenizer with a vocabulary size and the possibility to output embeddings of some dimension

              Args:
                lang (str): a language code name
                vocabulary_size (int): the size of the base vocabulary
                emb_size (int) : the size of the embeddings
                pad (str): string for padding tokens
                eos (str): string for eos token
                bos (str): string for bos token
        """
        super(BpeTokenizer, self).__init__('<unk>',pad,bos,eos)
        self.bpe = bpemb.BPEmb(lang=lang, vs=vocabulary_size, dim=emb_size)

        def replace_w2v_wf(toreplace,replaceby):
            idx = self.bpe.emb.key_to_index[toreplace]
            del self.bpe.emb.key_to_index[toreplace]
            self.bpe.emb.key_to_index[replaceby] = idx
            self.bpe.emb.index_to_key[idx] = replaceby

        if bos:
            self.bpe.BOS_str = bos
            replace_w2v_wf(self.bpe.BOS_str,bos)
        if eos:
            self.bpe.EOS_str = eos
            replace_w2v_wf(self.bpe.EOS_str, eos)

        self.vocabulary = self.bpe.emb.index_to_key + [pad]
        self.types2idx = {elt: idx for idx, elt in enumerate(self.vocabulary)}

        self.lang = lang
        self.vocabulary_size = vocabulary_size
        self.emb_size   = emb_size


    def tokenize(self, string):
        """
        Splits a string into tokens

        Args:
            string (str) : a string to tokenize

        Returns:
            a list of strings
        """
        if self._bos:
            tokens = [self.bos_token]
            tokens.extend(self.bpe.encode(string))
        else:
            tokens = self.bpe.encode(string)
        if self._eos:
            tokens.append(self.eos_token)
        return tokens


    def save_pretrained(self, dirpath):
        """
        Saves the tokenizer to model dir.

        Args:
            dirpath (path or string) : path to the tokenizer params dir
        """
        with open(os.path.join(dirpath, 'tokenizer.json'), 'w') as outfile:
            outfile.write(json.dumps({'pad': self.pad, 'lang':self.lang,'vs':self.vocabulary_size,'es':self.emb_size,
                                      'bos': self._bos, 'eos': self._eos, 'ClassName': self.__class__.__name__}))

    @staticmethod
    def from_pretrained(dirpath):
        """
        Loads the tokenizer from the model directory

        Args:
            dirpath (path or string) : path to the tokenizer params dir

            Returns:
                a BpeTokenizer object
        """
        with open(os.path.join(dirpath, 'tokenizer.json')) as infile:
            ldict = json.loads(infile.read())
            return BpeTokenizer(lang=ldict['lang'],
                                vocabulary_size=ldict['vs'],
                                emb_size=ldict['es'],
                                pad=ldict['pad'],
                                bos=ldict['bos'],
                                eos=ldict['eos'])


    def add_tokens(self, tokens):
        raise NotImplementedError('Attempt to modify the vocabulary of the BpeTokenizer. The vocabulary of this tokenizer cannot be modified.')




class PunctuationTokenizer(AbstractTokenizer):

  """
  Punctuation tokenizer that approximates the HuggingFace tokenizer interface.
  The tokenizer splits the input using punctuation. One may not be able to recover the original input after segmentation
  """
  def __init__(self, base_vocabulary=None, unk='<unk>',pad='<pad>',bos=None,eos=None):
      """
      Creates a tokenizer with some vocabulary and an unknown token

      Args:
        base_vocabulary (list): list of strings. If base vocabulary is None, the tokenizer is instanciated with a default list of 10000 word forms
        unk (str): string for unknown tokens
        pad (str): string for padding tokens
        eos (str): string for eos token
        bos (str): string for bos token
      """
      super(PunctuationTokenizer, self).__init__(unk,pad,bos,eos)

      if base_vocabulary is None:
          base_vocabulary = default_en_wordlist()
      assert(type(base_vocabulary) == list)
      self.add_tokens([self.unk,self.pad] + base_vocabulary + [ elt for elt in [bos,eos]  if elt is not None])

  @staticmethod
  def from_pretrained(dirpath):
      """
      Loads the tokenizer from the model directory

      Args:
        dirpath (path or string) : path to the tokenizer params dir

      Returns:
        a PunctuationTokenizer object
      """
      with open(os.path.join(dirpath, 'tokenizer.json')) as infile:
          ldict = json.loads(infile.read())
          return PunctuationTokenizer(ldict['vocabulary'], unk=ldict['unk'],
                                                           pad=ldict['pad'],
                                                           bos=ldict['bos'],
                                                           eos=ldict['eos'])

  def tokenize(self, string):
    """
    Splits a string into tokens

    Args:
      string (str) : a string to tokenize

    Returns:
      a list of strings
    """
    if self._bos:
      tokens = [self.bos_token]
      tokens.extend(string.split())
    else:
      tokens =  string.split()
    if self._eos:
      tokens.append(self.eos_token)
    return tokens



def all_en_inflections(wordform,pos):
    """
    Returns the inflections for an English lemma
    Args:
        wordform (str): an English lemma
        pos      (str): a part of speech code (n,v,a)
    Returns:
        a list of inflected wordforms including the original query lemma
    """
    result = [wordform]
    if pos == 'n':
        result.extend(__MORPH_GEN__.plural_noun(wordform))
    elif pos == 'v':
        result.extend(__MORPH_GEN__.plural_verb(wordform))
        if ' ' in wordform:
            elt,*rest = wordform.split()
            result.append(''.join([elt,'s',' ']+rest))
        else:
            result.append(wordform+"s")
    elif pos == 'a':
        result.extend(__MORPH_GEN__.plural_adj(wordform))
    return list(set(result))

class WordNetTokenizer(AbstractTokenizer):
    """
    This is a tokenizer that tokenizes text by recognizing explicitly the wordnet vocabulary.
    It may naturally recognize some multi-word expressions provided a properly normalized input text.
    It is possible to configure the tokenizer in a mode such that the exact input can be reconstructed from the output tokens.
    """
    def __init__(self,wordnet='oewn:2024', unk='<unk>', pad='<pad>', bos=None, eos=None):

        super(WordNetTokenizer, self).__init__(unk, pad,bos,eos)
        wn.download(wordnet)
        wordnet = wn.Wordnet(wordnet)

        wordlist = default_en_wordlist()

        #adds the wordnet vocabulary
        for w in wordnet.words():
            for wf in w.forms():
                wordlist.extend(all_en_inflections(wf,w.pos))
        self.wordnet = wordnet
        self.add_tokens([self.unk, self.pad] + wordlist + [elt for elt in [bos, eos] if elt is not None])
        self.vocabulary.sort(key=lambda x:len(x),reverse=True) #this is to achieve a longest match effect

    @staticmethod
    def from_pretrained(dirpath):
        """
        Loads the tokenizer from the model directory

        Args:
            dirpath (path or string) : path to the tokenizer params dir

        Returns:
            a WordNetTokenizer object
        """
        with open(os.path.join(dirpath, 'tokenizer.json')) as infile:
            ldict = json.loads(infile.read())
            return WordNetTokenizer(ldict['vocabulary'],
                                           unk=ldict['unk'],
                                           pad=ldict['pad'],
                                           bos=ldict['bos'],
                                           eos=ldict['eos'])

    def tokenize(self, string, include_separators = False,wn_lemmatizer=None):
        """
        Splits a string into tokens

        Args:
            string (str) : a string to tokenize
            include_separators (bool): flag controlling whether to include inter-token strings (whitespace most of the time)
            wn_lemmatizer (lemmatizer): A callable returning the wordnet lemma(s) for a given word string. wn.Morphy is an example

        Returns:
            a list of strings
        """
        if self._has_to_compile:
            self.tok_pattern     = re.compile('|'.join(re.escape(wf) for wf in self.vocabulary if not wf.isspace()))
            self._has_to_compile = False

        idxes = []
        for match in self.tok_pattern.finditer(string):
            idxes.append(match.span())

        if include_separators:
            idxes = list(set(i for span in idxes for i in span))
            idxes.sort()
            idxes = zip(idxes,idxes[1:])

        result = []
        if self._bos:
            result = [self.bos_token]
        for start,end in idxes:
            result.append(string[start:end])
        if self._eos:
            result.append(self.eos_token)
        if wn_lemmatizer:
            return [ wn_lemmatizer(token) for token in result]

        return result

    def add_tokens(self, tokens):
        """
        Adds a list of tokens to the vocabulary.

        Args:
            tokens :  a list of strings to add to the vocabulary
        """
        super().add_tokens(tokens)
        self._has_to_compile = True


if __name__ == '__main__':

    #quick how to
    test_sentence = "the boy's cat  sleeps   on the mat and the witch-hunter who had not a typhus fever doesn't care"

    print('Punctuation')
    tok = PunctuationTokenizer()
    print(tok.tokenize(test_sentence))

    print('\nLexical (Wordnet)')
    tok = WordNetTokenizer()
    print(tok.tokenize(test_sentence))

    tok = BpeTokenizer()
    print('\nBPE')
    print(tok.tokenize(test_sentence))
