from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags

import spacy
import pandas as pd


class Dataset:

    def __init__(self, 
                 path: str, 
                 verbose: bool = True) -> None:
        """ Can be used to load a dataset from a CSV file. The CSV file must have the following columns:
                - document: The document to be classified.
                - class_index: The class index of the document.
                - class_name: The class name of the document.
                - dataset_type: The type of the dataset. Can be 'train' or 'test'.

        Args:
            path (str): The path to the CSV file.
            verbose (bool, optional): Whether to print information about the dataset. Defaults to True.
        """
        self.path = path
        self.verbose = verbose

    def load(self) -> None:
        """ Loads the dataset from the CSV file. """
        self.dataset = pd.read_csv(self.path)

        self.dataset['tagged_docs'] = self.dataset.apply(
            lambda row: self.get_tagged_documents(row), axis=1
        )
        self.dataset['doc_key'] = self.dataset.index.astype(str)

        self.class_df = pd.DataFrame()
        self.class_df['class_index'] = self.dataset['class_index'].unique()
        self.class_df['class_name'] = self.class_df['class_index'].apply(
            lambda class_index: self.dataset[self.dataset['class_index'] == class_index]['class_name'].unique()[0]
        )
        keywords = self.get_spacy_similarity_keywords(self.dataset)
        self.class_df['keywords'] = self.class_df['class_name'].apply(
            lambda class_name: keywords[class_name]
        )
        self.class_df['number_of_keywords'] = self.class_df['keywords'].apply(lambda keywords: len(keywords))

    def get_tagged_documents(self, item: dict) -> TaggedDocument:
        """ Returns a TaggedDocument object from a dataset item.
        
        Args:
            item (dict): A dataset item.

        Returns:
            TaggedDocument: A TaggedDocument object.
        """
        return TaggedDocument(self.__tokenize(item['document']), [str(item.name)])

    def __tokenize(self, doc: str) -> list[str]:
        """ Tokenizes a document.
        
        Args:
            doc (str): The document to be tokenized.

        Returns:
            list[str]: The tokens extracted from the document.
        """
        return simple_preprocess(strip_tags(doc), deacc=True, min_len=2, max_len=15)
    
    def get_spacy_similarity_keywords(self, df: pd.DataFrame) -> dict:
        """ Returns a dictionary with the keywords for each class in the dataset using Spacy similarity.
        
        Args:
            df (pd.DataFrame): The dataset with the documents and classes.

        Returns:
            dict: A dictionary with the keywords for each class in the dataset using Spacy similarity.
        """
        classes_name = list(df['class_name'].unique())

        spacy_similarity_model = spacy.load('en_core_web_lg')
        resulting_keywords = {}

        all_documents_words = []
        for doc in df['document']:
            all_documents_words.extend(doc.split(' '))
        all_documents_words = list(set(all_documents_words))

        for class_name in classes_name:
            self.__print(f'[INFO] Processing class: {class_name}...')

            similarity_levels = []
            
            for word in all_documents_words:
                spacy_word_1 = spacy_similarity_model(word)
                spacy_word_2 = spacy_similarity_model(class_name)

                if(not (spacy_word_1 and spacy_word_1.vector_norm and spacy_word_2 and spacy_word_2.vector_norm)):
                    continue
                similarity_level = spacy_word_1.similarity(spacy_word_2)

                similarity_levels.append((word, similarity_level))

            similarity_levels = sorted(similarity_levels, key=lambda x: x[1], reverse=True)
            resulting_keywords[class_name] = [word for word, _ in similarity_levels]
            
        return resulting_keywords
    
    def __print(self, text: str) -> None:
        """ Prints a text if verbose is True.

        Args:
            text (str): The text to be printed.
        """
        if self.verbose:
            print(text)

    def get_document_dataframe(self) -> pd.DataFrame:
        """ Returns the documents inside the dataset as a DataFrame.

        Returns:
            pd.DataFrame: The documents inside the dataset as a DataFrame.
        """
        if hasattr(self, 'dataset'):
            return self.dataset
        else:
            raise Exception('Dataset not loaded. Call load() method first.')
        
    def get_class_dataframe(self) -> pd.DataFrame:
        """ Returns the classes inside the dataset as a DataFrame.

        Returns:
            pd.DataFrame: The classes inside the dataset as a DataFrame.
        """
        if hasattr(self, 'class_df'):
            return self.class_df
        else:
            raise Exception('Dataset not loaded. Call load() method first.')