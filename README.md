# Topic Modeling

Code relating to work carried out during an internship at [Sinch] (https://www.sinch.com/), consisting of an analysis of current topic modeling models. The report associated to the code can be found, as named "report.pdf".

## Description

Machine learning models require more and more data to be trained, particularly in the field of natural language processing (NLP). Manually labeling data takes up a considerable amount of time that could be devoted to a more useful task. One way of solving this problem is to use topic modeling to help us divide the data into clusters, and then label the data using these clusters as classes. This project evaluates several thematic modeling models such as LDA, [BERTopic](https://github.com/MaartenGr/BERTopic) and their guided version. It also presents results on unsupervised text classification.

## Getting Started

### Installing

The easiest way to run the notebooks contained in this repository is to create a conda environment using the yaml file provided:

```bash
conda env create -f environment.yml
```

For more information on conda, see the [documentation](https://conda.io/projects/conda/en/latest/index.html).

## Authors

* Yorick Estievenart ([Linkedin](https://be.linkedin.com/in/yorick-estievenart-634335248))

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* [OCTIS](https://github.com/MIND-Lab/OCTIS)
* [BERTopic](https://github.com/MaartenGr/BERTopic)
* [Lbl2Vec](https://github.com/sebischair/Lbl2Vec)
* [vONTSS](https://github.com/xuweijieshuai/vONTSS)

## Disclaimer

The findings were derived from a concise examination of existing topic models and should not be treated as definitive outcomes. It is advisable for any interested reader to peruse and replicate the experiments independently to verify the accuracy of the findings.