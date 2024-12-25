from torch.utils.data import Dataset


class RandomAccessRawText(Dataset):
    """
    Wraps a list of strings (sentences or paragraphs) into a pytorch dataset
    """
    #TODO implement a disk storage with memmapping kind of backend  to handle larger datasets

    def __init__(self,data,tokenizer):
        """
        Args:
            tokenizer (callable): a tokenizer is a callable mapping a string to a list of integers
        """
        self.data = [ tokenizer(chunk) for chunk in data ]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]