The Hub
########

The package provides basic utilities to access and retrieve small scale datasets and models
ready to use for in-class exercises. These utilities allow to download seamlessly standardized datasets
without dealing with messy urls and data cleanup as well as pretrained deep learning models. 

The package is oragnized around a hub of datasets that can be instanciated in python with the following
object ::

  from upnlp import HubUpc
  
  hub = HubUpc()

The hub contains two main types of data: **datasets** or data that can be used for training models and performing
predictions and **pretrained** models that are ready to be used in exercises. The content of the hub
can observed by calling the methods ::

  hub.list_datasets()
  hub.list_models()


Models
******
Tokenizers and models can be used seamlessly by using the `from_pretrained(path)` with a path in the hub. Such a path has the form `pretrained/<model_name>`. Thus one loads the tokenizer and a model named `zebra-5` in the hub using the command ::

  from upnlp import AutoTokenizer, AutoModel

  tokenizer = AutoTokenizer.from_pretrained('pretrained/zebra-5')
  model     = AutoModel.from_pretrained('pretrained/zebra-5')

to get the actual tokenizer and model class types, one can use the python `type`
function as ::

  type(tokenizer)
  type(model)

  
Datasets
********
Datasets can be explictly retrieved from the hub using the Hub method `get_local_path(path)` whose path is a dataset. To get the dataset named `shakespeare` one uses the command ::

  corpus_path = hub.get_local_path('datasets/shakespeare')

this downloads the dataset from the hub and stores it locally. It can be retrieved
at `corpus_path`.

Datasets can be used for training or evaluating models. Nowadays the data consumed by Deep NLP models takes the form of batches.
A batch is a matrix of integers coding the text sent to the model for inference. To transform the text in numeric matrices,
libraries provide convenience methods. For language modeling the `RandomAccessRawText` is a pytorch `Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ subclass that can be used by a pytorch `Dataloader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_. Here is an default way to instanciate a dataloader from a downloaded dataset stored in a string called *text*::

      from upnlp import normalize_text
      from torch.utils.data import DataLoader

      dataset     = nlp_datasets.RandomAccessRawText([normalize_text(sent) for sent in text.split('\n') if sent and not sent.isspace()],tokenizer)
      loader      = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=tokenizer.pad_batch)

The string is first split into chunks using newlines. The chunks are assembled in a dataset and tokenized with the `tokenizer`.
Once the dataset is built we use the standard pytorch interface to build the DataLoader used for generating batches of data.

Related API
***********

.. autoclass:: nlp_datasets::HubUpc
	       
.. autoclass:: nlp_datasets::RandomAccessRawText

