from model import DaCluDeK
import pandas as pd
from sklearn.metrics import f1_score



if __name__ == '__main__':
    defined_keywords = {}
    keywords_df = pd.read_csv(f'./cache/hblldrbtts_spacy_keywords.csv')
    
    for class_name in keywords_df['class_name'].unique():
        defined_keywords[class_name] = keywords_df[keywords_df['class_name'] == class_name]['class_result_keywords'].apply(lambda x: x[1:-1].replace('\'', '').split(', ')).tolist()[0][:10]

    documents_df = pd.read_csv('../../datasets/data/BBC_News/documents.csv')
    documents = documents_df['document'].tolist()#[:100]
    labels = documents_df['class_name'].tolist()#[:100]

    train_docs = documents_df[documents_df['dataset_type'] == 'train']['document'].tolist()#[:100]
    train_labels = documents_df[documents_df['dataset_type'] == 'train']['class_name'].tolist()#[:100]

    test_docs = documents_df[documents_df['dataset_type'] == 'test']['document'].tolist()#[:100]
    test_labels = documents_df[documents_df['dataset_type'] == 'test']['class_name'].tolist()#[:100]

    model = DaCluDeK(defined_keywords=defined_keywords, train_documents=documents, embedding_model_name='Doc2Vec', verbose=True, save_cache=True, load_cache=True)
    model.fit()

    y_pred_train = model.predict(train_docs)['class_name'].tolist()
    y_pred_test = model.predict(test_docs)['class_name'].tolist()

    f1_test = f1_score(test_labels, y_pred_test, average='micro')
    f1_train = f1_score(train_labels, y_pred_train, average='micro')

    print(f'F1 score on train set: {f1_train}')
    print(f'F1 score on test set: {f1_test}')

