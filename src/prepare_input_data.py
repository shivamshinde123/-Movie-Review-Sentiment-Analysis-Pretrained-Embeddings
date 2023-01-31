import logging, os, yaml
import string, re
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding
from get_data import GetData
from reading_params import ReadParams
import tensorflow_hub as hub


## creating a logger, file handler and the formatter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = ReadParams().read_params()

data_encoding_log_file_path = params['Log_paths']['data_encoding']

file_handler = logging.FileHandler(data_encoding_log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class DataEncoding:

    """The purpose of this class and its methods is to encode the text data

    Parameters
    -----------

    sequence_length : int
        Length of biggest sentence (default 300)
    vocab_size: int
        The size of vovabulary (default 20000)
    embedding_dim: int
        The size of output of embedding layer
    
    """

    def __init__(self):
        pass

    
    def create_embedding_layer(self):

        """This function is used to create the embeddings of the input integer encoded text

        Parameters
        -----------

        None

        Returns
        --------

        text_embedding: embedded vector for the input integer encoded data
        """

        try:
            embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
            hub_layer = hub.KerasLayer(embedding, input_shape=[], output_shape=[50], dtype=tf.string)

            logger.info('Imported pretrained embeddings for the input integer encoded data.')

        except Exception as e:
            logger.exception(e)
            raise e

        else:
            return hub_layer


    


        


    
