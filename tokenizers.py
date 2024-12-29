import torch
import json
import os
import hashlib
import nlp_datasets

def normalize_text(text):
    """
    For tokenizers splitting tokens on blank spaces it is relevant to normalize spacing before tokenization
    Args:
        text (str): the text to normalize
    Returns:
         the normalized text
    """
    translation_map = str.maketrans({'.':' . ','?':' ? ','!':' ! '})
    return text.translate(translation_map)


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
        return DefaultTokenizer.from_pretrained(local_path)


class DefaultTokenizer:

  """
  DefaultTokenizer that approximates the HuggingFace tokenizer interface.
  """

  def __init__(self, base_vocabulary, unk='<unk>',pad='<pad>',bos=None,eos=None):
      """
      Creates a tokenizer with some vocabulary and an unknown token

      Args:
        base_vocabulary: list of strings

      KwArgs:
        unk (str): string for unknown tokens
        pad (str): string for padding tokens
        eos (str): string for eos token
        bos (str): string for bos token
      """
      assert(type(base_vocabulary) == list)
      self.unk = unk
      self.pad = pad
      self._bos,self._eos = bos,eos
      self.vocabulary = []
      self.types2idx  = {}
      self.add_tokens([self.unk,self.pad] + base_vocabulary + [ elt for elt in [bos,eos]  if elt is not None])

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
    with open(os.path.join(dirpath,'tokenizer.json')) as infile:
      ldict = json.loads(infile.read())
      return DefaultTokenizer(ldict['vocabulary'],
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
    with open(os.path.join(dirpath,'tokenizer.json'),'w') as outfile:
      outfile.write(json.dumps({'unk':self.unk,'pad':self.pad,'vocabulary':self.vocabulary,'bos':self._bos,'eos':self._eos}))



  def add_tokens(self, tokens):
    """
    Adds a list of tokens to the vocabulary.

    Args:
      tokens :  a list of strings to add to the vocabulary
    """
    if not type(tokens) == list:
      raise Exception("Error tokens are not given as a list. Cannot continue anymore")

    for token in tokens:
      if token not in self.vocabulary:
        self.vocabulary.append(token)

    self.types2idx = {elt:idx for idx,elt in enumerate(self.vocabulary)}

  def tokenize(self, string):
    """
    Splits a string into tokens

    Args:
      string : a string to tokenize

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

  def convert_tokens_to_ids(self, tokens):
    """
    Maps a list of tokens to integer codes

    Args:
      tokens : a list of strings
    Returns:
      a list of integers
    """
    unkid = self.types2idx[self.unk]
    return [self.types2idx.get(token,unkid) for token in tokens]

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

  def decode(self,ids):
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


  def pad_batch(self,batch_codes):
    """
    Pads a batch of integers with the pad code

    Args:
      batch_codes : a list of lists of integers

    Returns:
      a list of lists of integers as a tensor
    """
    max_len      = max([len(sentence) for sentence in batch_codes])
    padded_codes = [ sentence + [self.pad_id]*(max_len-len(sentence)) for sentence in batch_codes]
    return torch.LongTensor(padded_codes)


if __name__ == '__main__':
    #quick how to
    tok = DefaultTokenizer(normalize_text("the cat sleeps on the mat?").split(),'<unk>',"<pad>",bos="<bos>",eos="<eos>")
    tok.save_pretrained("/tmp")
    tok = DefaultTokenizer.from_pretrained("/tmp")
