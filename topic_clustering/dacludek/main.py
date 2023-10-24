from model import DaCluDeK
import pandas as pd


if __name__ == '__main__':
    defined_keywords = {}
    keywords_df = pd.read_csv(f'./cache/cbsqppgcvm_spacy_keywords.csv')
    
    for class_name in keywords_df['class_name'].unique():
        defined_keywords[class_name] = keywords_df[keywords_df['class_name'] == class_name]['class_result_keywords'].apply(lambda x: x[1:-1].replace('\'', '').split(', ')).tolist()[0][:10]

    documents_df = pd.read_csv('../datasets/data/20NewsGroup/documents.csv')
    documents = documents_df['document'].tolist()#[:100]
    labels = documents_df['class_name'].tolist()#[:100]

    model = DaCluDeK(defined_keywords=defined_keywords, train_documents=documents, embedding_model_name='Doc2Vec', verbose=True, save_cache=True, load_cache=True)#, dimensionality_reduction_n_components=2)
    model.load_embeddings_model()