import nltk 
from collections import Counter
import numpy as np
import torch
import os
from tqdm import tqdm
from nltk.corpus import brown
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import sys

### As the code was inc it was becoming more and more confusing ki whats what like the datastr so Im changing the naming convention and will use this now:
## i== int , s== string, st == set , l== list, d== dict, t== tuple, f== function, c== class. for arrays I'll use p ==pointer , so for int arr will use ip, for string arr will use sp and so on. this will be added to the front of the variable name.
# 


## first load the brown corpus
nltk.download('brown')
nltk.download('stopwords')



# sentences = brown.sents()  ###i wanna see the data first so i can decide if it needs preprocessing -- edit: it does 
# print(sentences)



# -------------------------------------------------------------------------------------------#
# PREPROCESSING
# -------------------------------------------------------------------------------------------#

def f_preprocess(sp_sentences):

    # convert to lowercase
    sp_sentences = [[word.lower() for word in sent] for sent in sp_sentences]
    
    # remove punctuation and special characters
    sp_wsw_sentences = [[word for word in sent if word.isalnum()] for sent in sp_sentences] # wsw == with stop words
    
    # remove stop words
    st_sw = set(nltk.corpus.stopwords.words('english'))  # sw == stop words
    sp_woutsw_sentences = [[word for word in sent if word not in st_sw] for sent in sp_wsw_sentences] # woutsw == without stop words
    
    return sp_woutsw_sentences , sp_wsw_sentences



def f_build_vocab(sp_sentences, i_min_freq=5):
    sp_all_words = []
    for s_sent in sp_sentences:
        for s_word in s_sent:
            sp_all_words.append(s_word)
    
    d_word_freq = Counter(sp_all_words)

    # print(d_word_freq)
    
    # filter out words that appear less than min_freq times
    st_vocab = set()
    for s_word, i_freq in d_word_freq.items():
        if i_freq >= i_min_freq:
            st_vocab.add(s_word)

    # now will create a mapping of word to index and index to word for the vocab
    d_word_to_index = {word: idx for idx, word in enumerate(st_vocab)}
    d_index_to_word = {idx: word for word, idx in d_word_to_index.items()}

    return st_vocab, d_word_to_index, d_index_to_word


def f_build_cooccurrence_matrix(sp_sentences, d_word_to_index, i_window_size=2):

    i_vocab_size = len(d_word_to_index)
    # np_cooc_matrix = np.zeros((i_vocab_size, i_vocab_size), dtype=np.float32) # np == numpy, cooc == cooccurrence

    # for s_sent in tqdm(sp_sentences, desc="Building co-occurrence matrix"):
    #     for i, s_word in enumerate(s_sent):
    #         if s_word in d_word_to_index:
                
    #             word_idx = d_word_to_index[s_word]
                
    #             # look at the context window around the word
    #             for j in range(max(0, i - i_window_size), min(len(s_sent), i + i_window_size + 1)):
    #                 if j != i and s_sent[j] in d_word_to_index:  # skip the target word itself
    #                     context_idx = d_word_to_index[s_sent[j]]
    #                     np_cooc_matrix[word_idx][context_idx] += 1

    # return np_cooc_matrix

    np_cooc_matrix = lil_matrix((i_vocab_size, i_vocab_size), dtype=np.float32)

    for s_sent in tqdm(sp_sentences, desc="Building co-occurrence matrix"):
        for i, s_word in enumerate(s_sent):
            if s_word in d_word_to_index:
                word_idx = d_word_to_index[s_word]
                
                for j in range(max(0, i - i_window_size), min(len(s_sent), i + i_window_size + 1)):
                    if j != i and s_sent[j] in d_word_to_index:
                        context_idx = d_word_to_index[s_sent[j]]
                        np_cooc_matrix[word_idx, context_idx] += 1

    # Convert to CSR format for faster operations
    return np_cooc_matrix.tocsr()



def f_apply_svd(np_cooc_matrix, i_embedding_dim=100):
    """
    Apply SVD to the co-occurrence matrix to get word embeddings
    """

    print(f"  Matrix shape: {np_cooc_matrix.shape}")
    print(f"  Performing SVD decomposition (this may take a while)...")
    


    # Apply SVD: M = U * Sigma * V^T
    U, Sigma, VT = svds(np_cooc_matrix, k=i_embedding_dim)
    
    print(f"  SVD complete!")
    

    # Reduce to desired embedding dimension
    np_embeddings = U * np.sqrt(Sigma)
    
    print(f"  Final embedding shape: {np_embeddings.shape}")
    
    
    return np_embeddings



def f_find_similar_words(s_word, np_embeddings, d_word_to_index, d_index_to_word, i_top_k=5):
    """Find most similar words using cosine similarity"""
    if s_word not in d_word_to_index:
        print(f"Word '{s_word}' not in vocabulary")
        return
    
    word_idx = d_word_to_index[s_word]
    word_vec = np_embeddings[word_idx]
    
    # Compute norms
    embedding_norms = np.linalg.norm(np_embeddings, axis=1)
    word_norm = np.linalg.norm(word_vec)
    
    # Avoid division by zero - filter out zero vectors
    valid_indices = embedding_norms > 1e-10
    
    # Compute cosine similarity only for valid vectors
    similarities = np.zeros(len(np_embeddings))
    similarities[valid_indices] = np.dot(np_embeddings[valid_indices], word_vec) / (
        embedding_norms[valid_indices] * word_norm
    )
    
    # Get top k similar words (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1][1:i_top_k+1]
    
    print(f"\nMost similar words to '{s_word}':")
    for idx in top_indices:
        if not np.isnan(similarities[idx]):
            print(f"  {d_index_to_word[idx]}: {similarities[idx]:.4f}")



def f_save_embeddings(np_embeddings, d_word_to_index, s_filename):
    """
    Save embeddings and word mappings as .pt file
    """
    torch.save({
        'embeddings': torch.from_numpy(np_embeddings),
        'word_to_index': d_word_to_index
    }, s_filename)
    print(f"Embeddings saved to {s_filename}")



if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] in ['eval', '--evaluate']:
        print("\n=== EVALUATION MODE ===\n")
        
        # Load embeddings
        print("Loading embeddings...")
        woutsw_data = torch.load('embeddings/svd_woutsw.pt')
        wsw_data = torch.load('embeddings/svd_wsw.pt')
        
        np_woutsw_embeddings = woutsw_data['embeddings'].numpy()
        d_woutsw_word_to_index = woutsw_data['word_to_index']
        d_woutsw_index_to_word = {idx: word for word, idx in d_woutsw_word_to_index.items()}
        
        np_wsw_embeddings = wsw_data['embeddings'].numpy()
        d_wsw_word_to_index = wsw_data['word_to_index']
        d_wsw_index_to_word = {idx: word for word, idx in d_wsw_word_to_index.items()}
        
        print("Embeddings loaded!\n")
        
        # Test word similarities
        print("--- Testing WITHOUT stopwords embeddings ---")
        f_find_similar_words('king', np_woutsw_embeddings, d_woutsw_word_to_index, d_woutsw_index_to_word)
        f_find_similar_words('computer', np_woutsw_embeddings, d_woutsw_word_to_index, d_woutsw_index_to_word)
        f_find_similar_words('good', np_woutsw_embeddings, d_woutsw_word_to_index, d_woutsw_index_to_word)
        
        print("\n--- Testing WITH stopwords embeddings ---")
        f_find_similar_words('king', np_wsw_embeddings, d_wsw_word_to_index, d_wsw_index_to_word)
        f_find_similar_words('computer', np_wsw_embeddings, d_wsw_word_to_index, d_wsw_index_to_word)
        f_find_similar_words('good', np_wsw_embeddings, d_wsw_word_to_index, d_wsw_index_to_word)
        
    else:
        #----------------------- preprocessing --------------------------#
        sp_sentences = brown.sents()
        sp_woutsw_sentences, sp_wsw_sentences = f_preprocess(sp_sentences) # so woutsw == without stop words, wsw == with stop words loll   

        #----------------------- build vocab --------------------------#
        st_woutsw_vocab, d_woutsw_word_to_index, d_woutsw_index_to_word = f_build_vocab(sp_woutsw_sentences, i_min_freq=5)
        st_wsw_vocab, d_wsw_word_to_index, d_wsw_index_to_word = f_build_vocab(sp_wsw_sentences, i_min_freq=5)

        print("Vocabulary size without stop words: ", len(st_woutsw_vocab))
        print("Vocabulary size with stop words: ", len(st_wsw_vocab))

        # print(preprocessed_sentences[:10])

        #----------------------- build co-occurrence matrix --------------------------#
        print("\nBuilding co-occurrence matrix without stop words...")
        np_woutsw_cooc_matrix = f_build_cooccurrence_matrix(sp_woutsw_sentences, d_woutsw_word_to_index, i_window_size=10)
        
        print("Building co-occurrence matrix with stop words...")
        np_wsw_cooc_matrix = f_build_cooccurrence_matrix(sp_wsw_sentences, d_wsw_word_to_index, i_window_size=10)

        #----------------------- apply SVD --------------------------#
        i_embedding_dim = 100
        
        print(f"\nApplying SVD (embedding dim={i_embedding_dim}) without stop words...")
        np_woutsw_embeddings = f_apply_svd(np_woutsw_cooc_matrix, i_embedding_dim)
        
        print(f"Applying SVD (embedding dim={i_embedding_dim}) with stop words...")
        np_wsw_embeddings = f_apply_svd(np_wsw_cooc_matrix, i_embedding_dim)

        #----------------------- save embeddings --------------------------#
        os.makedirs('embeddings', exist_ok=True)
        f_save_embeddings(np_woutsw_embeddings, d_woutsw_word_to_index, 'embeddings/svd_woutsw.pt')
        f_save_embeddings(np_wsw_embeddings, d_wsw_word_to_index, 'embeddings/svd_wsw.pt')
        
        print("\nSVD embeddings training complete!")






