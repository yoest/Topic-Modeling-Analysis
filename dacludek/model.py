import spacy
import pandas as pd

from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class DaCluDeK:

    def __init__(self, 
                 defined_keywords: dict, 
                 train_documents: list,
                 embedding_model_name: str,
                 doc2vec_n_spacy_keywords: int = 20,
                 dimensionality_reduction_model_name: str = None,
                 dimensionality_reduction_n_components: int = -1,
                 load_cache: bool = False,
                 save_cache: bool = False,
                 verbose: bool = True) -> None:
        self.defined_keywords = defined_keywords
        self.train_documents = train_documents
        self.embedding_model_name = embedding_model_name
        self.dimensionality_reduction_model_name = dimensionality_reduction_model_name
        self.dimensionality_reduction_n_components = dimensionality_reduction_n_components
        self.doc2vec_n_spacy_keywords = doc2vec_n_spacy_keywords
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.verbose = verbose

        self.embedding_model = None
        self.dimensionality_reduction_model = None

    def fit(self):
        """ Fit the model on the training data using the defined keywords. """
        pass

    def __find_similar_documents_with_keywords(self) -> dict:
        """ Realise the third step of the algorithm which is to find similar documents with keywords by computing cosine similarity 
            between the encoded documents and encoded keywords.

        Returns:
            dict: Dictionary with class names as keys and list of tuples (document, similarity_score) as values.
        """
        pass

    def __clean_outliers(self) -> dict:
        """ Realise the fourth step of the algorithm which is to clean outliers by removing documents which are not similar to any 
            keyword by using an algorithm called Local Outlier Factor (LOF).
        
        Returns:
            dict: Dictionary with class names as keys and list of tuples (document, similarity_score) as values.
        """
        pass

    def __get_centroids_of_outlier_cleaned_documents(self) -> dict:
        """ Realise the fifth step of the algorithm which is to get the centroids of the outlier cleaned documents.

        Returns:
            dict: Dictionary with class names as keys and vector of the centroid of the outlier cleaned documents as values.
        """
        pass

    def classify(self, documents: list) -> pd.DataFrame:
        """ Classifies the documents.

        Args:
            documents (list): List of documents to be classified.

        Returns:
            pd.DataFrame: Dataframe with columns: document, class_name, similarity_score.
        """
        pass

    def load_embeddings_model(self):
        """ Loads the embeddings model. """
        if self.embedding_model_name == 'Doc2Vec':
            train_tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.train_documents)]
            if self.dimensionality_reduction_n_components != -1:
                self.embedding_model = Doc2Vec(train_tagged_documents, vector_size=self.dimensionality_reduction_n_components, window=2, min_count=1, workers=4)
            else:
                self.embedding_model = Doc2Vec(train_tagged_documents, window=2, min_count=1, workers=4)

            # Redefine the keywords with words that it has seen during the training to avoid random embeddings
            spacy_similarity_keywords_df = self.__get_spacy_similarity_keywords_for_documents(self.train_documents)
            for class_name in self.defined_keywords.keys():
                self.defined_keywords[class_name] = spacy_similarity_keywords_df[spacy_similarity_keywords_df['class_name'] == class_name]['class_result_keywords'].tolist()[0]

        elif self.embedding_model_name == 'SentenceTransformer':
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            if self.dimensionality_reduction_model_name == 'PCA':
                self.dimensionality_reduction_model = PCA(n_components=self.dimensionality_reduction_n_components)
                self.dimensionality_reduction_model.fit(self.embedding_model.encode(self.train_documents))
            
        else:
            raise Exception(f'Unknown embedding model name: {self.embedding_model_name}')
        
    def encode_documents(self, documents: list) -> list:
        """ Encodes the documents using the embedding model.

        Args:
            documents (list): List of documents.

        Returns:
            list: List of encoded documents.
        """
        if self.embedding_model_name == 'Doc2Vec':
            return [self.embedding_model.infer_vector(doc.split(' ')).tolist() for doc in documents]
        
        elif self.embedding_model_name == 'SentenceTransformer':
            if self.dimensionality_reduction_model_name == 'PCA':
                vectors = self.embedding_model.encode(documents)
                return self.dimensionality_reduction_model.transform(vectors).tolist()
            else:
                return self.embedding_model.encode(documents).tolist()
            
        else:
            raise Exception(f'Unknown embedding model name: {self.embedding_model_name}')
        
    def encode_keywords(self) -> dict:
        """ Encodes the keywords using the embedding model.

        Returns:
            dict: Dictionary with class names as keys and list of encoded keywords as values.
        """
        keywords_encoding = {}

        for class_name in self.defined_keywords.keys():
            keywords_encoding[class_name] = self.encode_documents(self.defined_keywords[class_name])

        return keywords_encoding

    def __get_spacy_similarity_keywords_for_documents(self, documents: list) -> dict:
        """ Doc2Vec model needs to knows the keywords on which it will be trained. This method returns the keywords for each class using the Spacy similarity model.

        Args:
            documents (list): List of documents.

        Returns:
            dict: Dictionary with class names as keys and list of keywords as values.
        """
        documents_repr = ''.join([item[0] for item in documents])[:10]
        if self.load_cache:
            cache_df = pd.read_csv(f'./cache/{documents_repr}_spacy_keywords.csv')
            cache_df['class_result_keywords'] = [item[:self.doc2vec_n_spacy_keywords] for item in cache_df['class_result_keywords'].apply(lambda x: x[1:-1].replace('\'', '').split(', '))]
            return cache_df

        spacy_similarity_model = spacy.load('en_core_web_lg')
        resulting_keywords = {}

        # Get all words from all documents as a vocabulary
        vocabulary = []
        for doc in documents:
            vocabulary.extend(doc.split(' '))
        vocabulary = list(set(vocabulary))

        for class_name in self.defined_keywords.keys():
            self.__print(f'[INFO] Processing keywords for class: {class_name}...' + ' ' * 50, end='\r')

            class_keywords = self.defined_keywords[class_name]
            similarity_levels_matrix = pd.DataFrame(columns=class_keywords, index=vocabulary)

            for word in vocabulary:
                spacy_word_1 = spacy_similarity_model(word)
                for class_keyword in class_keywords:
                    spacy_word_2 = spacy_similarity_model(class_keyword)

                    if(not (spacy_word_1 and spacy_word_1.vector_norm and spacy_word_2 and spacy_word_2.vector_norm)):
                        continue
                    similarity_level = spacy_word_1.similarity(spacy_word_2)

                    similarity_levels_matrix.loc[word, class_keyword] = similarity_level
            
            mean_by_word = similarity_levels_matrix.mean(axis=1)
            resulting_keywords[class_name] = [word for word in mean_by_word.sort_values(ascending=False).index]

        # Save the resulting keywords to a cache file
        cache_df = pd.DataFrame()
        cache_df['class_name'] = resulting_keywords.keys()
        cache_df['class_init_keywords'] = self.defined_keywords.values()
        cache_df['class_result_keywords'] = resulting_keywords.values()

        if self.save_cache:
            cache_df.to_csv(f'./cache/{documents_repr}_spacy_keywords.csv', index=False)

        cache_df['class_result_keywords'] = [item[:self.doc2vec_n_spacy_keywords] for item in cache_df['class_result_keywords']]
        return cache_df
    
    def __print(self, text: str, end: str = '\n'):
        if self.verbose:
            print(text, end=end)