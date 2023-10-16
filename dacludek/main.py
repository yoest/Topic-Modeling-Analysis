from model import DaCluDeK
import pandas as pd


if __name__ == '__main__':
    defined_keywords = {
        'business': ['business'],
        'entertainment': ['entertainment'],
        'politics': ['politics'],
        'sport': ['sport'],
        'tech': ['technology']
    }
    documents_df = pd.read_csv('../datasets/data/BBC_News/documents.csv')
    documents = documents_df['document'].tolist()#[:100]
    labels = documents_df['class_name'].tolist()#[:100]

    model = DaCluDeK(defined_keywords=defined_keywords, train_documents=documents, embedding_model_name='Doc2Vec', verbose=True, save_cache=True, load_cache=True, doc2vec_n_spacy_keywords=2)#, dimensionality_reduction_n_components=2)
    model.fit()
    # result = model.predict(documents)
    result = model.score(documents, labels)
    print(result)