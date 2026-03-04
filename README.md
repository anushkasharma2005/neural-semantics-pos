# Word Embeddings and POS Tagging - Assignment 2


## Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy nltk scikit-learn matplotlib seaborn tqdm pandas
```


- Download NLTK Data

```py
import nltk
nltk.download('brown')
nltk.download('stopwords')
nltk.download('universal_tagset')
```





## Running the Code


1. Train SVD Embeddings
```bash
python3 svd_embeddings.py
```


Outputs: svd_wsw.pt, svd_woutsw.pt

2. Train Word2Vec Embeddings
```bash
python3 word2vec.py
```
Outputs: embeddings/skipgram_w5_wsw.pt, embeddings/cbow_w5_wsw.pt

3. Train POS Tagger
```bash
python3 pos_tagger.py
```
Tests all embedding variants and saves models to models

4. Run Analogy Test
```bash
python3 task2.py
```
Outputs: analogy.txt

5. Run Bias Check
```bash
python3 task2_2.py
```
Outputs: bias.txt

6. Generate Visualizations
```bash
python3 plot.py
```
Outputs: 9 PNG files in `graphs/`


### Pre-trained Embeddings

Download GloVe embeddings:

```bash

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir glove_6B
mv glove.6B.100d.txt glove_6B/

```


`*Key finding: Including stopwords improved accuracy by 26-31% across all models.*`

okay so all the extra folders and files are uplodede in the one drive, can access them from here 
https://iiithydresearch-my.sharepoint.com/:f:/g/personal/anushka_sharma_research_iiit_ac_in/IgAgv5GqVb2MSaH8RZgsYw0PAae-ou2F39I8xDpYHWRCql8?e=3y7q17

