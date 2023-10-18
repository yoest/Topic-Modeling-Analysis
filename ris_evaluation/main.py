from evaluator import Evaluator

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd


if __name__ == '__main__':
    documents_df = pd.read_csv('./datasets/data/BBC_News/documents.csv')
    documents_df = documents_df

    documents = documents_df['document'].tolist()

    vectorizer = CountVectorizer()
    documents = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(documents)

    # Get the words for each found topic as list
    topics = []
    for topic in lda.components_:
        topic_words = []
        for i in topic.argsort()[-10:]:
            topic_words.append(vectorizer.get_feature_names_out()[i])
        topics.append(topic_words)

    output = {
        "topics": topics,
        "topic-document-matrix": None,
        "topic-word-matrix": None,
        "test-topic-document-matrix": None
    }
    
    evaluator = Evaluator(model_output=output)

    # -- Coherence
    coherence = evaluator.compute_coherence()
    print(coherence)

    # -- Diversity
    diversity = evaluator.compute_diversity()
    print(diversity)

    # -- Supervised correlation
    predicted_topics = lda.transform(documents)
    predicted_topics = predicted_topics.argmax(axis=1)

    words_by_extracted_topics = {}
    for idx, topic in enumerate(topics):
        words = documents_df.iloc[idx]['document'].split()

        if idx not in words_by_extracted_topics:
            words_by_extracted_topics[idx] = {}

        for word in words:
            if word not in words_by_extracted_topics[idx]:
                words_by_extracted_topics[idx][word] = 0

            words_by_extracted_topics[idx][word] += 1

    true_classes = documents_df['class_name'].tolist()

    words_by_class = {}
    for idx, class_name in enumerate(true_classes):
        words = documents_df.iloc[idx]['document'].split()

        if class_name not in words_by_class:
            words_by_class[class_name] = {}

        for word in words:
            if word not in words_by_class[class_name]:
                words_by_class[class_name][word] = 0

            words_by_class[class_name][word] += 1

    supervised_correlation = evaluator.compute_supervised_correlation(words_by_extracted_topics, words_by_class)
    print(supervised_correlation)