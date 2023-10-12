from model import DaCluDeK
import pandas as pd


if __name__ == '__main__':
    defined_keywords = {
        'business': ['business', 'company'],
        'entertainment': ['entertainment'],
        'politics': ['politics', 'constitution'],
        'sport': ['sport'],
        'tech': ['technology']
    }
    documents = pd.read_csv('../datasets/data/BBC_News/documents.csv')
    documents = documents['document'].tolist()[:100]

    model = DaCluDeK(defined_keywords=defined_keywords, train_documents=documents, embedding_model_name='Doc2Vec', verbose=True, save_cache=True, load_cache=False, doc2vec_n_spacy_keywords=3, dimensionality_reduction_model_name = 'PCA', dimensionality_reduction_n_components = 2)
    model.load_embeddings_model()
    # print(model.encode_documents(documents))
    # print(model.encode_keywords())