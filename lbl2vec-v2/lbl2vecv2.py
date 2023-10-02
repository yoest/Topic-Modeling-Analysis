from lbl2vec import Lbl2Vec, Lbl2TransformerVec
from dataset import Dataset
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd


class Lbl2VecV2:

    def __init__(self, 
                 dataset: Dataset, 
                 use_transformer: bool = False,
                 n_keywords: int = 10,
                 n_iterations: int = 5,
                 verbose: bool = True) -> None:
        """ Can be used to train and evaluate a Lbl2VecV2 model (using the Lbl2Vec or 
            Lbl2TransformerVec architecture) on a dataset.

        Args:
            dataset (Dataset): The dataset to be used.
            use_transformer (bool, optional): Whether to use the Lbl2TransformerVec architecture. Defaults to False.
            n_keywords (int, optional): The number of keywords to be used for each class. Defaults to 10.
            n_iterations (int, optional): The number of iterations to be used for the evaluation. Defaults to 5.
            verbose (bool, optional): Whether to print information about the model. Defaults to True.
        """
        self.dataset = dataset
        self.dataset_df = self.dataset.get_document_dataframe()
        self.class_df = self.dataset.get_class_dataframe()

        self.use_transformer = use_transformer
        self.n_keywords = n_keywords
        self.n_iterations = n_iterations
        self.verbose = verbose

    def fit(self) -> None:
        """ Fits the model on the dataset. """
        if self.use_transformer:
            self.model = Lbl2TransformerVec(
                keywords_list=[keywords[:self.n_keywords] for keywords in self.class_df['keywords']],
                documents=self.dataset_df['document'][self.dataset_df['dataset_type'] == 'train'],
                label_names=list(self.class_df['class_name']),
                verbose=self.verbose
            )
        else:
            self.model = Lbl2Vec(
                keywords_list=[keywords[:self.n_keywords] for keywords in self.class_df['keywords']],
                tagged_documents=self.dataset_df['tagged_docs'][self.dataset_df['dataset_type'] == 'train'], 
                label_names=list(self.class_df['class_name']),
                min_count=2,
                verbose=self.verbose
            )

        self.model.fit()

    def predict(self, documents: list[str]) -> str:
        """ Predicts the class of a list of documents.

        Args:
            documents (list[str]): A list of documents.

        Returns:
            str: A DataFrame containing the predicted class of each document.
        """
        if self.use_transformer:
            return self.model.predict_new_docs(documents=documents)
        else:
            documents_transformed = pd.DataFrame(documents, columns=['document'])
            tagged_documents = []
            for _, row in documents_transformed.iterrows():
                tagged_documents.append(self.dataset.get_tagged_documents(row))
            return self.model.predict_new_docs(tagged_docs=tagged_documents)
        
    def predict_train_documents(self) -> str:
        """ Predicts the class of the train documents.

        Returns:
            str: A DataFrame containing the predicted class of each train document.
        """
        if self.use_transformer:
            return self.predict(self.dataset_df['document'][self.dataset_df['dataset_type'] == 'train'])
        else:
            return self.model.predict_model_docs()
        
    def predict_test_documents(self) -> str:
        """ Predicts the class of the test documents.
        
        Returns:
            str: A DataFrame containing the predicted class of each test document.
        """
        return self.predict(self.dataset_df['document'][self.dataset_df['dataset_type'] == 'test'])
    
    def evaluate(self, average: str = 'micro') -> dict:
        """ Evaluates the model on the dataset i.e. calculates the F1 score for the train and test documents.

        Args:
            average (str, optional): The type of averaging to be used for the F1 score. Defaults to 'micro'.

        Returns:
            dict: A dictionary containing the F1 score for the train and test documents.
        """
        if self.use_transformer:
            return self.__evaluate_with_transformer(average=average)
        else:
            return self.__evaluate_without_transformer(average=average)
    
    def __evaluate_without_transformer(self, average: str) -> dict:
        """ Evaluates the model on the dataset without using the Lbl2TransformerVec architecture.

        Args:
            average (str, optional): The type of averaging to be used for the F1 score. Defaults to 'micro'.

        Returns:
            dict: A dictionary containing the F1 score for the train and test documents.
        """
        avg_train_f1_scores = []
        avg_test_f1_scores = []

        for _ in range(self.n_iterations):
            train_docs_lbl_similarities = self.predict_train_documents()
            test_docs_lbl_similarities = self.predict_test_documents()

            evaluation_train = train_docs_lbl_similarities.merge(self.dataset_df[self.dataset_df['dataset_type'] == 'train'], left_on='doc_key', right_on='doc_key')
            evaluation_test = test_docs_lbl_similarities.merge(self.dataset_df[self.dataset_df['dataset_type'] == 'test'], left_on='doc_key', right_on='doc_key')

            y_true_train = evaluation_train['class_name']
            y_pred_train = evaluation_train['most_similar_label']

            y_true_test = evaluation_test['class_name']
            y_pred_test = evaluation_test['most_similar_label']

            f1_scores_train = f1_score(y_true_train, y_pred_train, average=average)
            f1_scores_test = f1_score(y_true_test, y_pred_test, average=average)

            avg_train_f1_scores.append(f1_scores_train)
            avg_test_f1_scores.append(f1_scores_test)

        return np.mean(avg_train_f1_scores), np.mean(avg_test_f1_scores)
    
    def __evaluate_with_transformer(self, average: str) -> dict:
        """ Evaluates the model on the dataset using the Lbl2TransformerVec architecture.

        Args:
            average (str, optional): The type of averaging to be used for the F1 score. Defaults to 'micro'.

        Returns:
            dict: A dictionary containing the F1 score for the train and test documents.
        """
        train_docs_lbl_similarities = self.predict_train_documents()
        test_docs_lbl_similarities = self.predict_test_documents()

        evaluation_train = train_docs_lbl_similarities.merge(self.dataset_df[self.dataset_df['dataset_type'] == 'train'], left_on='doc_key', right_on='doc_key')
        evaluation_test = test_docs_lbl_similarities.merge(self.dataset_df[self.dataset_df['dataset_type'] == 'test'], left_on='doc_key', right_on='doc_key')

        y_true_train = evaluation_train['class_name']
        y_pred_train = evaluation_train['most_similar_label']

        y_true_test = evaluation_test['class_name']
        y_pred_test = evaluation_test['most_similar_label']

        f1_scores_train = f1_score(y_true_train, y_pred_train, average=average)
        f1_scores_test = f1_score(y_true_test, y_pred_test, average=average)

        return f1_scores_train, f1_scores_test