import os
import requests
from torch.utils.data import Dataset
from pathlib import Path
import zipfile

__all__ = ['HubUpc','RandomAccessRawText']


__HUB_ROOT__ = os.path.join(Path.home(),'upnlp_hub')


class HubUpc:
    """
    This is a client allowing to download datasets and pretrained models from UPC nextcloud
    """
    def __init__(self):
        self.local_hub_root = __HUB_ROOT__

        self.datasets_db = {'shakespeare':'https://cloud.parisdescartes.fr/index.php/s/Ao5YqWsfz49mf9M',
                            'ulysses':'https://cloud.parisdescartes.fr/index.php/s/s7RXQQDecxEgKay',
                            'wikitext103':"https://cloud.parisdescartes.fr/index.php/s/pwN63TkKL5QtCqi",
                            'opensubtitles':'https://cloud.parisdescartes.fr/index.php/s/aqCzgGF5diTrf8A',
                            'twitter':'https://cloud.parisdescartes.fr/index.php/s/5zymCK68D8gC4Ky',
                            'semcor':'https://cloud.parisdescartes.fr/index.php/s/o67jnk4gTw3SndD',
                            'enwords':'https://cloud.parisdescartes.fr/index.php/s/FyikmpFEMM2EePL'}

        self.models_db   = {'zebra-lstm':''}

        if not os.path.exists(self.local_hub_root):
            os.mkdir(self.local_hub_root)
            os.mkdir(os.path.join(self.local_hub_root,'pretrained'))
            os.mkdir(os.path.join(self.local_hub_root,'datasets'))



    def list_datasets(self):
        """
        Returns a list of all dataset identifiers currently in the hub

        Returns:
            a list of strings
        """
        return list(self.datasets_db.keys())

    def list_models(self):
        """
        Returns a list of all pretrained models identifiers currently in the hub

        Returns:
            a list of strings
        """
        return list(self.models_db.keys())


    def get_local_path(self,path):
        """
        Returns the local path for a dataset or a model if it comes from the hub
        or the path itself if it is not in the hub

        Args:
            path (path) : a path
        Returns:
            path
        """
        fullpath = os.path.join(self.local_hub_root,path)
        if os.path.exists(fullpath):
            return fullpath

        chunks = os.path.split(path)
        if len(chunks) != 2:
            return path
        head,tail = chunks
        if head == 'datasets':
            dic = self.datasets_db
        elif head == "pretrained":
            dic = self.models_db
        else:
            return path

        if tail in dic:
            self._download_dir(tail,chunk_type=head)
            return fullpath
        else:
            return path


    def _download_dir(self, chunk_name,chunk_type='datasets'):
        """
        Downloads a full chunk from UPC nextcloud and puts it in cache if not already cached

        Args:
           chunk_name (str): the name of the dataset
        KwArgs:
           chunk_type (str): either 'datasets' or 'pretrained'
        """
        if chunk_type == 'datasets' and chunk_name not in self.datasets_db:
            raise Exception("Error. this dataset is not available. Use list_datasets() to get an up to date list of available data sets")

        if chunk_type == 'pretrained' and chunk_name not in self.models_db:
            raise Exception("Error. this model is not available. Use list_models() to get an up to date list of available models")

        chunk_path = os.path.join(self.local_hub_root,chunk_type, chunk_name)
        if not os.path.exists(chunk_path):
            os.mkdir(chunk_path)
            print(f'Downloading files for {chunk_name}...')
            db = self.datasets_db if chunk_type == 'datasets' else self.models_db
            remotefile = db[chunk_name]
            localfile  = f'{chunk_path}.zip'
            localpath  = os.path.join(self.local_hub_root,chunk_type,localfile)
            self._download_binary_file(remotefile, localpath)
            with zipfile.ZipFile(localpath,'r') as dzip:
                dzip.extractall(os.path.join(self.local_hub_root,chunk_type))
            os.remove(localpath)


            print(f'Files written in {chunk_path}')



    def _download_binary_file(self,hubfilename,destination_path):
        """
        Downloads a single binary file from the nextcloud

        Args:
            hubfilename (str)       : name of the shared file on UPC NextCloud
            destination_path (path) : path where to write the file
        """
        result = requests.get(f'{hubfilename}/download',stream=True)
        with open(destination_path, 'wb') as outstream:
            for chunk in result.iter_content(chunk_size=1024):
                outstream.write(chunk)



    def _download_text_file(self,hubfilename,destination_path):
        """
        Downloads a single text file from the nextcloud
        Args:
            hubfilename (str)       : name of the shared file on UPC NextCloud
            destination_path (path) : path where to write the file
        """
        result = requests.request(method='get',url=f'{hubfilename}/download')
        if result.status_code != 200:
            raise Exception(f'Warning I could not download textfile {hubfilename}, not written to {destination_path}')
        with open(destination_path,'w') as outstream:
            outstream.write(result.text)




class RandomAccessRawText(Dataset):
    """
    Wraps a list of strings (sentences or paragraphs) into a pytorch dataset.
    Tokenized sequences that are longer than some size are truncated and split into several data items
    """
    def __init__(self,data,tokenizer,max_seq_size=512):
        """
        Args:
            data (list): a list of strings (sentences or paragraphs)

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

if __name__ == '__main__':
    hub = HubUpc()
    print(hub.list_datasets())
    hub.download_dir('shakespeare',chunk_type='datasets')
    print(hub.list_models())
    hub.download_dir('zebra-lstm',chunk_type='pretrained')

