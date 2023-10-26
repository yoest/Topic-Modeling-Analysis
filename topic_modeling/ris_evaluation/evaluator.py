from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

import pandas as pd


class Evaluator:

    def __init__(self, model_output: dict) -> None:
        """ Can be use to evaluate Topic Modeling models.

        Args:
            model_output (dict): The output of a trained model. This should be on the OCTIS format (see https://github.com/MIND-Lab/OCTIS).
        """
        self.model_output = model_output

    def compute_coherence(self) -> dict:
        """ Compute the coherence of the model.

        Returns:
            dict: A dictionary containing the coherence of the model (using the 'c_v', 'c_uci', 'c_npmi' and 'u_mass' measures).
        """
        c_v = Coherence(measure='c_v')
        c_uci = Coherence(measure='c_uci')
        c_npmi = Coherence(measure='c_npmi')
        u_mass = Coherence(measure='u_mass')

        return {
            'c_v': c_v.score(self.model_output),
            'c_uci': c_uci.score(self.model_output),
            'c_npmi': c_npmi.score(self.model_output),
            'u_mass': u_mass.score(self.model_output)
        }
    
    def compute_diversity(self) -> float:
        """ Compute the diversity of the model.

        Returns:
            float: The diversity of the model.
        """
        diversity = TopicDiversity()
        return diversity.score(self.model_output)
    
    def compute_supervised_correlation(self, words_by_extracted_topics: dict, words_by_class: dict) -> float:
        """ Compute the supervised correlation between the words of the extracted topics and the words of the classes.

        Args:
            words_by_extracted_topics (dict): The words of the extracted topics.
            words_by_class (dict): The words of the classes.

        Returns:
            float: The supervised correlation between the words of the extracted topics and the words of the classes.
        """
        avg_scores = []

        for extracted_topic in words_by_extracted_topics.keys():
            highest_score = []

            for class_name in words_by_class.keys():
                words_for_extracted_topic = words_by_extracted_topics[extracted_topic]
                words_for_class = words_by_class[class_name]

                score = self.__compute_single_correlation(words_for_extracted_topic, words_for_class)
                highest_score.append(score)

            highest_score = max(highest_score)
            avg_scores.append(highest_score)

        return sum(avg_scores) / len(avg_scores)

        
    def __compute_single_correlation(self, words_for_extracted_topic: dict, words_for_class: dict) -> float:
        """ Compute the correlation between the words of a topic and the words of a class.

        Args:
            words_for_extracted_topic (dict): The words of the topic.
            words_for_class (dict): The words of the class.

        Returns:
            float: The correlation between the words of a topic and the words of a class.
        """
        resulting_score = 0
        total_count = 0

        for topic_word in words_for_extracted_topic:
            count_topic_word = words_for_extracted_topic[topic_word]
            count_class_word = words_for_class.get(topic_word, 0)
            total_count += count_topic_word + count_class_word

            score = 1 - abs((count_topic_word - count_class_word) / max(count_topic_word, count_class_word))
            score = score * (count_topic_word + count_class_word)

            resulting_score += score

        return resulting_score / total_count