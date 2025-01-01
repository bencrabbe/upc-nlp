import torch
import json
import os
import hashlib
import re
import wn
import inflect
import nlp_datasets


__all__ = ['normalize_text','AutoTokenizer','WordNetTokenizer','PunctuationTokenizer']


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


__MORPH_GEN__ = inflect.engine()

class AutoTokenizer:
    """
    This is a namespace for easy loading of tokenizers defined from this module.
    """
    @staticmethod
    def from_pretrained(path):
        """
        Loads a pretrained model from directory.

        Args:
            path(path) : path to the model dir (either in the hub or local)

        Returns:
            the pretrained model instance
        """
        hub = nlp_datasets.HubUpc()
        local_path = hub.get_local_path(path)
        return PunctuationTokenizer.from_pretrained(local_path)




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

    @staticmethod
    def from_pretrained(dirpath):
        """
        Loads the tokenizer from the model directory

        Args:
          dirpath (path or string) : path to the tokenizer params dir

        Returns:
          a DefaultTokenizer object
        """
        with open(os.path.join(dirpath, 'tokenizer.json')) as infile:
            ldict = json.loads(infile.read())
            if ldict['ClassName'] == 'PunctuationTokenizer':
                return PunctuationTokenizer(ldict['vocabulary'],unk=ldict['unk'],
                                                                pad=ldict['pad'],
                                                                bos=ldict['bos'],
                                                                eos=ldict['eos'])
            elif ldict['ClassName'] == 'WordNetTokenizer':
                return WordNetTokenizer(__WORDNET__,ldict['vocabulary'],
                                                    unk=ldict['unk'],
                                                    pad=ldict['pad'],
                                                    bos=ldict['bos'],
                                                    eos=ldict['eos'])

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





class PunctuationTokenizer(AbstractTokenizer):

  """
  Punctuation tokenizer that approximates the HuggingFace tokenizer interface.
  The tokenizer splits the input using punctuation. One may not be able to recover the original input after segmentation
  """
  def __init__(self, base_vocabulary, unk='<unk>',pad='<pad>',bos=None,eos=None):
      """
      Creates a tokenizer with some vocabulary and an unknown token

      Args:
        base_vocabulary (list): list of strings
        unk (str): string for unknown tokens
        pad (str): string for padding tokens
        eos (str): string for eos token
        bos (str): string for bos token
      """
      assert(type(base_vocabulary) == list)
      super(PunctuationTokenizer, self).__init__(unk,pad,bos,eos)
      self.add_tokens([self.unk,self.pad] + base_vocabulary + [ elt for elt in [bos,eos]  if elt is not None])



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

        #downloads a basic English wordlist (functional words are generally not in wordnet)
        hub  = nlp_datasets.HubUpc()
        path = os.path.join(hub.get_local_path('datasets/enwords'),'enwordlist.txt')
        with open(path) as infile:
            wordlist = infile.read().split()

        #put the wordnet vocabulary
        for w in wordnet.words():
            for wf in w.forms():
                wordlist.extend(all_en_inflections(wf,w.pos))
        self.wordnet = wordnet
        self.add_tokens([self.unk, self.pad] + wordlist + [elt for elt in [bos, eos] if elt is not None])
        self.vocabulary.sort(key=lambda x:len(x),reverse=True) #this is to achieve a longest match effect

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
    tok = PunctuationTokenizer(normalize_text("the cat sleeps on the mat?").split(),'<unk>',"<pad>",bos="<bos>",eos="<eos>")
    tok.save_pretrained("/tmp")
    tok = PunctuationTokenizer.from_pretrained("pretrained/zebra")
    print(tok.tokenize("the boy's cat  sleeps   on the mat and the witch-hunter who had not a typhus fever doesn't care"))

    tok = WordNetTokenizer(__WORDNET__,'<unk>',"<pad>",bos="<bos>",eos="<eos>")
    print(tok.tokenize("the boy's cat  sleeps   on the mat and the witch-hunter who had not a typhus fever doesn't care"))

