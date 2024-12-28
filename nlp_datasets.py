from torch.utils.data import Dataset


class RandomAccessRawText(Dataset):
    """
    Wraps a list of strings (sentences or paragraphs) into a pytorch dataset.
    This dataset does not preserve any order between data items
    """

    def __init__(self,data,tokenizer,max_seq_size=512):
        """
        Args:
            tokenizer (callable): a tokenizer is a callable mapping a string to a list of integers

        KwArgs:
           max_seq_size (int): maximum size of a data sequence. Longer sequences are truncated
        """
        self.data = [  trunc_chunk for chunk in data for trunc_chunk in self._truncate(tokenizer(chunk),max_seq_size)]


    def _truncate(self,seq,size):

        while len (seq) > size:
            yield seq[:size]
            seq = seq[size:]

        if seq:
            yield seq


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]