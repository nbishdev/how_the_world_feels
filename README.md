# How the world feels
Query News API for a topic and perform sentiment analysis.

## Setup Instructions
### Using **pip**
```console
python3 -m venv virtualenv-world
source virtualenv-world/bin/activate
pip install -r ./requirements.txt
```
### Using **conda**
```console
python3 -m venv virtualenv-world
conda create -n virtualenv-world
conda activate virtualenv-world
conda install -c conda-forge requests nltk textblob stanza scikit-learn wordcloud -y
pip install eng_spacysentiment
```

## Execution Instructions
```console
usage: main.py [-h] -k KEY -t TOPIC [-s] [-a {nltk,textblob,spacy,corenlp}] [-y YEAR]

optional arguments:
  -h, --help            show this help message and exit
  -k KEY, --key KEY     News API key
  -t TOPIC, --topic TOPIC
                        topic string to query from News API
  -s, --sentiment       perform sentiment analysis
  -a {nltk,textblob,spacy,corenlp}, --analyzer {nltk,textblob,spacy,corenlp}
                        sentiment analyzer method
  -y YEAR, --year YEAR  article temporal restriction by year
```
