Language modeling
#################

This educational package provides basic language modeling facilities
for a wider audience of students with limited programming skills.

There are three families of language models available:

* Markovian language models
* Recurrent language models
* Transformer language models

To use a pretrained language model in python, please load both its tokenizer and its parameters
within python. Here is how to load a markovian (5-gram) language model trained on Shakespeare plays::

  from upnlp import PunctuationTokenizer, MarkovianLM

  tokenizer = PunctuationTokenizer.from_pretrained('pretrained/shakespeare-5')
  model     = MarkovianLM.from_pretrained('pretrained/shakespeare-5')

The **tokenizer** naturally splits strings into tokens. To observe how the tokenizer actually
performs the segmentation, one calls ::

  tokenizer.tokenize("Hamlet lives in Denmark")

the tokenizer also maps a string to a list if integer-coded tokens. This is achieved by calling ::

  tokenizer("Hamlet lives in Denmark")

and finally the tokenizer can map a list of integers back to a readable string ::

  tokenizer.decode([12,2,19,20,179])

The **model** can then be used to generate text by completing a prompt: ::

  model.generate(tokenizer('Hamlet lives in'), eos_token_id=tokenizer.eos_token_id)

The model can also be used to predict the log probabilities :math:`P(w_i| \text{prefix})` for each token
in a sequence :math:`w_1 ... w_n` ::

  model.logprobs(tokenizer('Hamlet lives in Denmark'))

and finally one can evaluate the model on a dataset to get its perplexity ::

  model.validate(dataloader,perplexity=True)

Related API
===========

.. autoclass:: tokenizers::PunctuationTokenizer
                :inherited-members:

.. autoclass:: lm_models::MarkovianLM
                :inherited-members: Module

.. autoclass:: lm_models::LstmLM
                :inherited-members: Module

.. autoclass:: lm_models::TransformerLM
                :inherited-members: Module




  










