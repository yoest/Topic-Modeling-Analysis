# Sinch - Topic Modelling/Clustering

### To-Do

- Find other embedding than Word2Vec, all-MiniLM-L6-v2, ...
- Observe the results when two topics are similar (lbl2vec limitations).
- Redo the experiments with simple clustering (TF-IDF + KMeans).
- Evaluate the performances of the Lbl2TransformerVec model with different embeddings.
- Provide an in-depth analysis of the assignment confidence on the DLBP and M10 datasets.

### In Progress

- Try to answer to this question: **Why are the F1 scores so bad?**. Possible explanation:
    - Choice of keywords at the first step not appropriate to find the classes?
    - Choice of embeddings?
- Create a super simple benchmark dataset with 2-3 classes to make an in-depth analysis of the keywords.

### Done

- ~~Get the limitations of the model.~~
- ~~Redo the roadmap and organise observations in two parts (with full input and reduced input).~~
- ~~Fix the issue with italic text slide 23.~~
- ~~Understand why the basic Lbl2Vec model is random and how to choose the number of iterations to compute an average.~~
- ~~Get informations about how spaCy similarity works.~~
- ~~Observe the probability of assigning a document to a class.~~
- ~~See the impact of the number of keywords choosen for the Doc2Vec-based model with spaCy and how this number varies with each datasets.~~
- ~~Evaluate the performances of the Lbl2Vec model.~~
