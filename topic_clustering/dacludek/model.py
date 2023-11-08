import spacy
import pandas as pd
import numpy as np
import os

from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from gensim.utils import simple_preprocess

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class DaCluDeK:

    def __init__(self, 
                 defined_keywords: dict, 
                 train_documents: list,
                 embedding_model_name: str,
                 doc2vec_n_spacy_keywords: int = 20,
                 dimensionality_reduction_model_name: str = None,
                 dimensionality_reduction_n_components: int = -1,
                 similarity_threshold: float = 0.5,
                 min_similarity_doc_count: int = 20,
                 load_cache: bool = False,
                 save_cache: bool = False,
                 verbose: bool = True) -> None:
        self.defined_keywords = defined_keywords.copy()
        self.train_documents = train_documents.copy()
        self.embedding_model_name = embedding_model_name
        self.dimensionality_reduction_model_name = dimensionality_reduction_model_name
        self.dimensionality_reduction_n_components = dimensionality_reduction_n_components
        self.similarity_threshold = similarity_threshold
        self.min_similarity_doc_count = min_similarity_doc_count
        self.doc2vec_n_spacy_keywords = doc2vec_n_spacy_keywords
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.verbose = verbose

        self.embedding_model = None
        self.dimensionality_reduction_model = None
        self.class_centroids = None

        self.cache_path = os.path.dirname(os.path.realpath(__file__)) + '/cache'

    def fit(self):
        """ Fit the model on the training data using the defined keywords.

        Raises:
            Exception: If the class centroids keys are not equal to defined keywords keys.
        """
        self.load_embeddings_model()

        embedded_train_documents = self.encode_documents(self.train_documents)
        embedded_keywords = self.encode_keywords()

        similar_documents = self.__find_similar_documents_with_keywords(embedded_train_documents, embedded_keywords)
        cleaned_similar_documents = self.__clean_outliers(similar_documents)
        self.class_centroids = self.__get_centroids_of_outlier_cleaned_documents(cleaned_similar_documents)

        if self.class_centroids.keys() != self.defined_keywords.keys():
            raise Exception(f'Class centroids keys ({self.class_centroids.keys()}) are not equal to defined keywords keys ({self.defined_keywords.keys()})')

    def __find_similar_documents_with_keywords(self, embedded_docs: list, embedded_keywords: dict) -> dict:
        """ Realise the third step of the algorithm which is to find similar documents with keywords by computing cosine similarity 
            between the encoded documents and encoded keywords.

        Args:
            embedded_docs (list): List of embedded documents.
            embedded_keywords (dict): Dictionary with class names as keys and list of encoded keywords as values.
        
        Returns:
            dict: Dictionary with class names as keys and list of tuples (document, similarity_score) as values.
        """
        similar_documents = {}

        for doc_embedding in embedded_docs:
            # Compute cosine similarity between the document and each keyword
            for class_name in embedded_keywords.keys():
                cosine_similarity = [self.__compute_cosine_similarity(doc_embedding, keyword_embedding) for keyword_embedding in embedded_keywords[class_name]]
                cosine_similarity = np.mean(cosine_similarity)

                similar_documents.setdefault(class_name, []).append((doc_embedding, cosine_similarity))

        for class_name in similar_documents.keys():
            similar_documents[class_name] = sorted(similar_documents[class_name], key=lambda x: x[1], reverse=True)
            
            count_beq_threshold = len([item for item in similar_documents[class_name] if item[1] >= self.similarity_threshold])
            count_beq_threshold = count_beq_threshold if count_beq_threshold >= self.min_similarity_doc_count else self.min_similarity_doc_count

            similar_documents[class_name] = similar_documents[class_name][:count_beq_threshold]

        return similar_documents

    def __compute_cosine_similarity(self, a: list, b: list) -> float:
        """ Computes the cosine similarity between two vectors (between 0 and 1)

        Args:
            a (list): Vector a.
            b (list): Vector b.

        Returns:
            float: Cosine similarity between a and b.
        """
        cosine_between_one_minus_one = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosine_between_zero_one = (cosine_between_one_minus_one + 1) / 2
        return cosine_between_zero_one

    def __clean_outliers(self, full_similar_document: dict) -> dict:
        """ Realise the fourth step of the algorithm which is to clean outliers by removing documents which are not similar to any 
            keyword by using an algorithm called Local Outlier Factor (LOF).
        
        Args:
            full_similar_document (dict): Dictionary with class names as keys and list of tuples (document, similarity_score) as values.
        
        Returns:
            dict: Dictionary with class names as keys and list of tuples (document, similarity_score) as values.
        """
        cleaned_similar_documents = {}

        for class_name in full_similar_document.keys():
            self.__print(f'[INFO] Cleaning outliers for class: {class_name}...' + ' ' * 50, end='\r')

            documents = [item[0] for item in full_similar_document[class_name]]
            outliers = LocalOutlierFactor(n_neighbors=20).fit_predict(documents)
            cleaned_similar_documents[class_name] = [item for i, item in enumerate(full_similar_document[class_name]) if outliers[i] == 1]

        return cleaned_similar_documents

    def __get_centroids_of_outlier_cleaned_documents(self, cleaned_similar_document: dict) -> dict:
        """ Realise the fifth step of the algorithm which is to get the centroids of the outlier cleaned documents.

        Args:
            cleaned_similar_document (dict): Dictionary with class names as keys and list of tuples (document, similarity_score) as values.

        Returns:
            dict: Dictionary with class names as keys and vector of the centroid of the outlier cleaned documents as values.
        """
        centroids = {}

        for class_name in cleaned_similar_document.keys():
            self.__print(f'[INFO] Computing centroids for class: {class_name}...' + ' ' * 50, end='\r')

            documents = [item[0] for item in cleaned_similar_document[class_name]]
            centroids[class_name] = np.stack(documents).mean(axis=0).tolist()

        return centroids

    def predict(self, documents: list) -> pd.DataFrame:
        """ Classifies the documents.

        Args:
            documents (list): List of documents to be classified.

        Returns:
            pd.DataFrame: Dataframe with columns: document, class_name, similarity_score.
        """
        self.__print(f'[INFO] Classifying documents...' + ' ' * 50, end='\r')

        embedded_documents = self.encode_documents(documents)

        classification_results = []
        for i, document in enumerate(embedded_documents):
            cosine_similarity = {}

            # Compute cosine similarity between the document and each class centroid
            for class_name in self.class_centroids.keys():
                cosine_similarity[class_name] = self.__compute_cosine_similarity(document, self.class_centroids[class_name])

            # Get the class name with the highest similarity score
            class_name = max(cosine_similarity, key=cosine_similarity.get)
            classification_results.append([documents[i], class_name, cosine_similarity[class_name]])

        return pd.DataFrame(classification_results, columns=['document', 'class_name', 'similarity_score'])
    
    def score(self, documents: list, labels: list) -> float:
        """ Computes the accuracy, f1-score, precision and recall of the classification.

        Args:
            documents (list): List of documents to be classified.
            labels (list): List of labels for the documents.

        Returns:
            float: Accuracy of the classification.
        """
        predictions = self.predict(documents)
        return self.score_with_predictions(predictions, labels)
    
    def score_with_predictions(self, predictions: pd.DataFrame, labels: list) -> pd.DataFrame:
        """ Computes the accuracy, f1-score, precision and recall of the classification.

        Args:
            predictions (pd.DataFrame): Dataframe with columns: document, class_name, similarity_score.
            labels (list): List of labels for the documents.

        Returns:
            float: Accuracy of the classification.
        """
        y_pred = predictions['class_name'].tolist()

        accuracy = accuracy_score(labels, y_pred)
        f1 = f1_score(labels, y_pred, average='macro')
        precision = precision_score(labels, y_pred, average='macro')
        recall = recall_score(labels, y_pred, average='macro')

        return pd.DataFrame([[accuracy, f1, precision, recall]], columns=['accuracy', 'f1', 'precision', 'recall'])

    def load_embeddings_model(self):
        """ Loads the embeddings model. """
        if self.embedding_model_name == 'Doc2Vec':
            train_tagged_documents = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(self.train_documents)]

            doc2vec_args = {
                'documents': train_tagged_documents,
                'window': 2,
                'min_count': 1,
                'workers': 4,
                'epochs': 10,
                'sample': 1e-5,
                'negative': 5,
                'hs': 1,
                'dm': 0,
                'dbow_words': 1
            }
            self.embedding_model = Doc2Vec(**doc2vec_args)

            # These are not needed since the __init__ method is already doing this
            # self.embedding_model.build_vocab(train_tagged_documents)
            # self.embedding_model.train(train_tagged_documents, total_examples=self.embedding_model.corpus_count, epochs=self.embedding_model.epochs)

            if self.dimensionality_reduction_model_name == 't-SNE':
                self.dimensionality_reduction_model = TSNE(n_components=self.dimensionality_reduction_n_components, method='exact')

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
            vectors = [self.embedding_model.infer_vector(doc.split(' ')).tolist() for doc in documents]

            if self.dimensionality_reduction_model_name == 't-SNE':
                return self.dimensionality_reduction_model.fit_transform(vectors)
            
            return vectors
        
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
            cache_df = pd.read_csv(f'{self.cache_path}/{documents_repr}_spacy_keywords.csv')
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
            cache_df.to_csv(f'{self.cache_path}/{documents_repr}_spacy_keywords.csv', index=False)

        cache_df['class_result_keywords'] = [item[:self.doc2vec_n_spacy_keywords] for item in cache_df['class_result_keywords']]
        return cache_df
    
    def __print(self, text: str, end: str = '\n'):
        if self.verbose:
            print(text, end=end)