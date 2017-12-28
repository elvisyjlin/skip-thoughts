# Skip-Thought Vectors

This is an migrated implementation of Skip-Thoughts for Python 3 from [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/skip_thoughts).

Skip thoughts model is a powerful sentence encoder, which encodes sentences with Seq2Seq model. 
The meanings of sentences are kept in the encoded vectors through this method. 
However, the code mentioned above does not support Python3.

Due to the lack of compatibility with Python3, I made some modifications to the original implementations. 
And I write up a simple way to encode sentences with combined skip thought vectors.

The original design is described in:  
Jamie Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
Antonio Torralba, Raquel Urtasun, Sanja Fidler.
[Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf).
*In NIPS, 2015.*


## Code author

Original code author: Chris Shallue ([@cshallue](https://github.com/cshallue))

Code editor: Elvis Yu-Jing Lin ([@elvisyjlin](https://github.com/elvisyjlin))


## How to use?

### Get the pre-trained models

To download the pre-trained models on the [BookCorpus](http://yknzhu.wixsite.com/mbweb) dataset.

```bash
bash get_skip_thoughts_pretrained_models.sh
```

### Encode sentences

To encode sentences into combined skip-thought vectors (unidirectional + bidirectional):

Please go to ```encode-by-skip-thoughts.ipynb``` and put you data loader in.

Or follow the instructions in [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/skip_thoughts):

```python
# Encode sentences with unidirectional skip thought vectors

import numpy as np
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

# TODO: Load your dataset here.
data = []

VOCAB_FILE = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=False),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

encodings = encoder.encode(data)
```
