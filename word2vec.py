import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import brown
import sys
import os

# Download required NLTK data
nltk.download('brown')
nltk.download('stopwords')

# -------------------------------------------------------------------------------------------#
# HYPERPARAMETERS
# -------------------------------------------------------------------------------------------#

EMBEDDING_DIM = 100
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 1024
MIN_WORD_FREQ = 3

# -------------------------------------------------------------------------------------------#
# PREPROCESSING (reuse from SVD)
# -------------------------------------------------------------------------------------------#

def f_preprocess(sp_sentences):
    """Preprocess sentences: lowercase, remove punctuation, filter stopwords"""
    
    # convert to lowercase
    sp_sentences = [[word.lower() for word in sent] for sent in sp_sentences]
    
    # remove punctuation and special characters
    sp_wsw_sentences = [[word for word in sent if word.isalnum()] for sent in sp_sentences]
    
    # remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    sp_woutsw_sentences = [[word for word in sent if word not in stop_words] for sent in sp_wsw_sentences]
    
    return sp_woutsw_sentences, sp_wsw_sentences


def f_build_vocab(sp_sentences, i_min_freq=5):
    """Build vocabulary with frequency filtering"""
    
    # flatten all sentences and count word frequencies
    all_words = []
    for sent in sp_sentences:
        for word in sent:
            all_words.append(word)
    
    word_freq = Counter(all_words)
    
    # filter out words that appear less than min_freq times
    vocab = set()
    for word, freq in word_freq.items():
        if freq >= i_min_freq:
            vocab.add(word)
    
    # create word to index mapping
    word_to_idx = {}
    idx = 0
    for word in sorted(vocab):
        word_to_idx[word] = idx
        idx += 1
    
    # create index to word mapping
    idx_to_word = {}
    for word, idx in word_to_idx.items():
        idx_to_word[idx] = word
    
    return vocab, word_to_idx, idx_to_word


# -------------------------------------------------------------------------------------------#
# DATA GENERATION FOR SKIP-GRAM
# -------------------------------------------------------------------------------------------#

def f_generate_skipgram_data(sp_sentences, d_word_to_idx, i_window_size):
    """
    Generate training pairs for Skip-Gram
    Returns: list of (target_word_idx, context_word_idx) pairs
    """
    training_pairs = []
    
    for sent in sp_sentences:
        # filter out words not in vocab
        sent_indices = [d_word_to_idx[word] for word in sent if word in d_word_to_idx]
        
        for i, target_idx in enumerate(sent_indices):
            # get context window
            start = max(0, i - i_window_size)
            end = min(len(sent_indices), i + i_window_size + 1)
            
            for j in range(start, end):
                if j != i:
                    context_idx = sent_indices[j]
                    training_pairs.append((target_idx, context_idx))
    
    return training_pairs


# -------------------------------------------------------------------------------------------#
# DATA GENERATION FOR CBOW
# -------------------------------------------------------------------------------------------#

def f_generate_cbow_data(sp_sentences, d_word_to_idx, i_window_size):
    """
    Generate training pairs for CBOW
    Returns: list of (context_word_indices, target_word_idx) pairs
    """
    training_data = []
    
    for sent in sp_sentences:
        # filter out words not in vocab
        sent_indices = [d_word_to_idx[word] for word in sent if word in d_word_to_idx]
        
        for i, target_idx in enumerate(sent_indices):
            # get context window
            start = max(0, i - i_window_size)
            end = min(len(sent_indices), i + i_window_size + 1)
            
            context_indices = []
            for j in range(start, end):
                if j != i:
                    context_indices.append(sent_indices[j])
            
            if len(context_indices) > 0:
                training_data.append((context_indices, target_idx))
    
    return training_data


# -------------------------------------------------------------------------------------------#
# SKIP-GRAM MODEL
# -------------------------------------------------------------------------------------------#

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        
        # target word embeddings (input)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # context word embeddings (output)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # initialize with small random values
        self.target_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
    
    def forward(self, target_words, context_words, negative_words):
        """
        target_words: (batch_size,)
        context_words: (batch_size,)
        negative_words: (batch_size, num_negative_samples)
        """
        
        # get embeddings
        target_embeds = self.target_embeddings(target_words)  # (batch_size, embedding_dim)
        context_embeds = self.context_embeddings(context_words)  # (batch_size, embedding_dim)

        negative_embeds = self.context_embeddings(negative_words)  # (batch_size, neg_samples, embedding_dim)
        
        # positive score
        positive_score = torch.sum(target_embeds * context_embeds, dim=1)  # (batch_size,)
        positive_loss = -torch.log(torch.sigmoid(positive_score))
        
        # negative scores
        negative_score = torch.bmm(negative_embeds, target_embeds.unsqueeze(2)).squeeze(2)  # (batch_size, neg_samples)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)  # (batch_size,)
        
        # total loss
        loss = torch.mean(positive_loss + negative_loss)
        
        return loss
    
    def get_embeddings(self):
        """Return the learned target embeddings"""
        return self.target_embeddings.weight.data.cpu().numpy()


# -------------------------------------------------------------------------------------------#
# CBOW MODEL
# -------------------------------------------------------------------------------------------#

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        
        # context word embeddings (input)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # target word embeddings (output)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # initialize with small random values
        self.context_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.target_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
    
    # def forward(self, context_words, target_words, negative_words):
    #     """
    #     context_words: (batch_size, context_size) - variable context size
    #     target_words: (batch_size,)
    #     negative_words: (batch_size, num_negative_samples)
    #     """
        
    #     # get context embeddings and average them
    #     context_embeds = self.context_embeddings(context_words)  # (batch_size, context_size, embedding_dim)
    #     context_embeds = torch.mean(context_embeds, dim=1)  # (batch_size, embedding_dim)
        
    #     if context_mask is not None:
    #         context_mask = context_mask.unsqueeze(2)  # (batch_size, context_size, 1)
    #         context_embeds = context_embeds * context_mask
    #         context_embeds = torch.sum(context_embeds, dim=1) / torch.sum(context_mask, dim=1)
    #     else:
    #         context_embeds = torch.mean(context_embeds, dim=1)



    #     # get target and negative embeddings
    #     target_embeds = self.target_embeddings(target_words)  # (batch_size, embedding_dim)
    #     negative_embeds = self.target_embeddings(negative_words)  # (batch_size, neg_samples, embedding_dim)
        
    #     # positive score
    #     positive_score = torch.sum(context_embeds * target_embeds, dim=1)  # (batch_size,)
    #     positive_loss = -torch.log(torch.sigmoid(positive_score))
        
    #     # negative scores
    #     negative_score = torch.bmm(negative_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # (batch_size, neg_samples)
    #     negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)  # (batch_size,)
        
    #     # total loss
    #     loss = torch.mean(positive_loss + negative_loss)
        
    #     return loss


    def forward(self, context_words, context_mask, target_words, negative_words):
        """
        context_words: (batch_size, context_size) - variable context size
        context_mask: (batch_size, context_size) - mask for valid context words
        target_words: (batch_size,)
        negative_words: (batch_size, num_negative_samples)
        """
        
        # get context embeddings
        context_embeds = self.context_embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        
        # apply mask and average only valid context words
        context_mask_expanded = context_mask.unsqueeze(2)  # (batch_size, context_size, 1)
        context_embeds = context_embeds * context_mask_expanded
        context_embeds = torch.sum(context_embeds, dim=1) / torch.sum(context_mask_expanded, dim=1)
        
        # get target and negative embeddings
        target_embeds = self.target_embeddings(target_words)  # (batch_size, embedding_dim)
        negative_embeds = self.target_embeddings(negative_words)  # (batch_size, neg_samples, embedding_dim)
        
        # positive score
        positive_score = torch.sum(context_embeds * target_embeds, dim=1)  # (batch_size,)
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10)
        
        # negative scores
        negative_score = torch.bmm(negative_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # (batch_size, neg_samples)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score) + 1e-10), dim=1)  # (batch_size,)
        
        # total loss
        loss = torch.mean(positive_loss + negative_loss)
        
        return loss
    

    
    
    def get_embeddings(self):
        """Return the learned context embeddings"""
        return self.context_embeddings.weight.data.cpu().numpy()


# -------------------------------------------------------------------------------------------#
# NEGATIVE SAMPLING
# -------------------------------------------------------------------------------------------#

def f_create_negative_sampling_distribution(word_freq, d_word_to_idx):
    """Create probability distribution for negative sampling (word^0.75)"""
    
    vocab_size = len(d_word_to_idx)
    word_freqs = np.zeros(vocab_size)
    
    for word, idx in d_word_to_idx.items():
        word_freqs[idx] = word_freq.get(word, 1)
    
    # raise to power 0.75 (as in original Word2Vec paper)
    word_freqs = np.power(word_freqs, 0.75)
    
    # normalize
    word_freqs = word_freqs / np.sum(word_freqs)
    
    return word_freqs


def f_sample_negatives(batch_size, num_negative, negative_dist, device):
    """Sample negative words for a batch"""
    
    negative_samples = np.random.choice(
        len(negative_dist),
        size=(batch_size, num_negative),
        p=negative_dist
    )
    
    return torch.LongTensor(negative_samples).to(device)


# -------------------------------------------------------------------------------------------#
# TRAINING FUNCTION
# -------------------------------------------------------------------------------------------#

def f_train_model(model, training_data, negative_dist, epochs, batch_size, learning_rate, device, model_type='skipgram'):
    """Train Skip-Gram or CBOW model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Add learning rate scheduler (reduces LR every 5 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)



    print(f"\nTraining {model_type.upper()} model...")
    print(f"  Total training pairs: {len(training_data)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # shuffle training data
        np.random.shuffle(training_data)
        
        # create batches
        for i in tqdm(range(0, len(training_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = training_data[i:i+batch_size]
            
            if model_type == 'skipgram':
                # unpack skip-gram batch
                target_words = torch.LongTensor([pair[0] for pair in batch]).to(device)
                context_words = torch.LongTensor([pair[1] for pair in batch]).to(device)
                
            else:  # CBOW
                # unpack CBOW batch - need to handle variable context sizes
                context_words_list = [pair[0] for pair in batch]
                target_words = torch.LongTensor([pair[1] for pair in batch]).to(device)
                
                # pad context words to same length
                max_context_len = max(len(ctx) for ctx in context_words_list)
                context_words = torch.zeros((len(batch), max_context_len), dtype=torch.long).to(device)
                context_mask = torch.zeros((len(batch), max_context_len), dtype=torch.float32).to(device)
                
                for j, ctx in enumerate(context_words_list):
                    context_words[j, :len(ctx)] = torch.LongTensor(ctx)
                    context_mask[j, :len(ctx)] = 1.0  # Mark valid positions
            # sample negative words
            negative_words = f_sample_negatives(len(batch), NEGATIVE_SAMPLES, negative_dist, device)
            
            # forward pass
            optimizer.zero_grad()
            
            if model_type == 'skipgram':
                loss = model(target_words, context_words, negative_words)
            else:  # CBOW
                loss = model(context_words, context_mask, target_words, negative_words)
            
            # backward pass
            loss.backward()
            # Add gradient clipping to prevent CBOW collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Step the scheduler
        scheduler.step()


    print("Training complete!")
    return model


# -------------------------------------------------------------------------------------------#
# SAVE EMBEDDINGS
# -------------------------------------------------------------------------------------------#

def f_save_embeddings(np_embeddings, d_word_to_idx, s_filepath):
    """Save embeddings to file"""
    
    torch.save({
        'embeddings': torch.FloatTensor(np_embeddings),
        'word_to_index': d_word_to_idx
    }, s_filepath)
    
    print(f"Embeddings saved to {s_filepath}")


# -------------------------------------------------------------------------------------------#
# EVALUATION
# -------------------------------------------------------------------------------------------#

def f_find_similar_words(s_word, np_embeddings, d_word_to_index, d_index_to_word, i_top_k=5):
    """Find most similar words using cosine similarity"""
    if s_word not in d_word_to_index:
        print(f"Word '{s_word}' not in vocabulary")
        return
    
    word_idx = d_word_to_index[s_word]
    word_vec = np_embeddings[word_idx]
    
    # compute norms
    embedding_norms = np.linalg.norm(np_embeddings, axis=1)
    word_norm = np.linalg.norm(word_vec)
    
    # avoid division by zero
    valid_indices = embedding_norms > 1e-10
    
    # compute cosine similarity
    similarities = np.zeros(len(np_embeddings))
    similarities[valid_indices] = np.dot(np_embeddings[valid_indices], word_vec) / (
        embedding_norms[valid_indices] * word_norm
    )
    
    # get top k similar words (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1][1:i_top_k+1]
    
    print(f"\nMost similar words to '{s_word}':")
    for idx in top_indices:
        if not np.isnan(similarities[idx]):
            print(f"  {d_index_to_word[idx]}: {similarities[idx]:.4f}")


# -------------------------------------------------------------------------------------------#
# MAIN
# -------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    
    # check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # check mode
    if len(sys.argv) > 1 and sys.argv[1] in ['eval', '--evaluate']:
        print("\n=== EVALUATION MODE ===\n")
        
        # ask which model to evaluate
        print("Which model do you want to evaluate?")
        print("  1. Skip-Gram")
        print("  2. CBOW")
        print("  3. Both")
        choice = input("Enter choice (1/2/3): ").strip()
        
        models_to_eval = []
        if choice == '1':
            models_to_eval = ['skipgram']
        elif choice == '2':
            models_to_eval = ['cbow']
        else:
            models_to_eval = ['skipgram', 'cbow']
        
        # evaluate selected models
        for model_type in models_to_eval:
            model_path = f'embeddings/{model_type}_window_{WINDOW_SIZE}_wsw.pt'
            
            if not os.path.exists(model_path):
                print(f"\nModel '{model_type}_window_{WINDOW_SIZE}_wsw.pt' not found. Train it first!")
                continue
            
            print(f"\n{'='*70}")
            print(f"EVALUATING {model_type.upper()}")
            print(f"{'='*70}")
            
            # load embeddings
            print(f"Loading embeddings from {model_path}...")
            data = torch.load(model_path)
            np_embeddings = data['embeddings'].numpy()
            d_word_to_idx = data['word_to_index']
            d_idx_to_word = {idx: word for word, idx in d_word_to_idx.items()}
            
            print(f"Embeddings loaded! Shape: {np_embeddings.shape}")
            
            # test word similarities
            print(f"\n--- Testing {model_type.upper()} embeddings ---")
            f_find_similar_words('king', np_embeddings, d_word_to_idx, d_idx_to_word)
            f_find_similar_words('computer', np_embeddings, d_word_to_idx, d_idx_to_word)
            f_find_similar_words('good', np_embeddings, d_word_to_idx, d_idx_to_word)
            f_find_similar_words('man', np_embeddings, d_word_to_idx, d_idx_to_word)
            f_find_similar_words('woman', np_embeddings, d_word_to_idx, d_idx_to_word)
    else:
        print("\n=== TRAINING MODE ===\n")
        
        # load and preprocess data
        print("Loading and preprocessing Brown corpus...")
        sp_sentences = brown.sents()
        sp_woutsw_sentences, sp_wsw_sentences = f_preprocess(sp_sentences)
        
        # build vocabulary
        print("Building vocabulary...")
        st_vocab, d_word_to_idx, d_idx_to_word = f_build_vocab(sp_wsw_sentences, i_min_freq=MIN_WORD_FREQ)
        word_freq = Counter([word for sent in sp_wsw_sentences for word in sent if word in d_word_to_idx])
        
        print(f"Vocabulary size: {len(st_vocab)}")
        
        # ask which model to train
        print("\nWhich model do you want to train?")
        print("  1. Skip-Gram")
        print("  2. CBOW")
        print("  3. Both")
        choice = input("Enter choice (1/2/3): ").strip()
        
        models_to_train = []
        if choice == '1':
            models_to_train = ['skipgram']
        elif choice == '2':
            models_to_train = ['cbow']
        else:
            models_to_train = ['skipgram', 'cbow']
        
        # train selected models
        for model_type in models_to_train:
            print(f"\n{'='*70}")
            print(f"TRAINING {model_type.upper()}")
            print(f"{'='*70}")
            
            # generate training data
            if model_type == 'skipgram':
                print("Generating Skip-Gram training pairs...")
                training_data = f_generate_skipgram_data(sp_wsw_sentences, d_word_to_idx, WINDOW_SIZE)
            else:
                print("Generating CBOW training data...")
                training_data = f_generate_cbow_data(sp_wsw_sentences, d_word_to_idx, WINDOW_SIZE)
            
            # create negative sampling distribution
            negative_dist = f_create_negative_sampling_distribution(word_freq, d_word_to_idx)
            
            # create model
            if model_type == 'skipgram':
                model = SkipGram(len(st_vocab), EMBEDDING_DIM)
            else:
                model = CBOW(len(st_vocab), EMBEDDING_DIM)
            
            # train model
            model = f_train_model(
                model, training_data, negative_dist, 
                EPOCHS, BATCH_SIZE, LEARNING_RATE, device, model_type
            )
            
            # extract embeddings
            embeddings = model.get_embeddings()
            
            # save embeddings
            os.makedirs(f'embeddings/{model_type}/', exist_ok=True)
            f_save_embeddings(embeddings, d_word_to_idx, f'embeddings/{model_type}/{model_type}_window_{WINDOW_SIZE}_wsw.pt')
            
            # quick test
            print(f"\n--- Quick Test for {model_type.upper()} ---")
            f_find_similar_words('king', embeddings, d_word_to_idx, d_idx_to_word)
            f_find_similar_words('computer', embeddings, d_word_to_idx, d_idx_to_word)
            f_find_similar_words('good', embeddings, d_word_to_idx, d_idx_to_word)
        
        print("\n" + "="*70)
        print("All models trained successfully!")
        print("="*70)