import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import random
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Download required NLTK data
nltk.download('brown')
nltk.download('universal_tagset')

# -------------------------------------------------------------------------------------------#
# HYPERPARAMETERS
# -------------------------------------------------------------------------------------------#

EMBEDDING_TYPE = 'skipgram'  # Options: 'svd', 'skipgram', 'cbow', 'glove'
EMBEDDING_PATH = 'embeddings/skipgram/skipgram_window_5.pt'  # Path to embeddings
CONTEXT_WINDOW = 2  # Number of words on each side to consider
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64
DROPOUT = 0.3

# -------------------------------------------------------------------------------------------#
# LOAD DATASET
# -------------------------------------------------------------------------------------------#

def f_load_brown_corpus():
    """Load Brown corpus with universal POS tags"""
    
    print("Loading Brown corpus with universal tagset...")
    
    # Get tagged sentences
    tagged_sents = brown.tagged_sents(tagset='universal')
    
    print(f"Total sentences: {len(tagged_sents)}")
    
    return tagged_sents


def f_split_data(tagged_sents, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data into train/val/test sets"""
    
    # Convert to list and shuffle sentences
    random.seed(42)
    shuffled = list(tagged_sents)  # Convert ConcatenatedCorpusView to list
    random.shuffle(shuffled)
    
    # Calculate split indices
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} sentences ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} sentences ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} sentences ({len(test_data)/total*100:.1f}%)")
    
    return train_data, val_data, test_data


# -------------------------------------------------------------------------------------------#
# BUILD VOCABULARIES
# -------------------------------------------------------------------------------------------#

def f_build_tag_vocab(tagged_sents):
    """Build tag vocabulary from tagged sentences"""
    
    all_tags = []
    for sent in tagged_sents:
        for word, tag in sent:
            all_tags.append(tag)
    
    unique_tags = sorted(set(all_tags))
    
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    print(f"\nPOS Tags ({len(unique_tags)}):")
    print(f"  {unique_tags}")
    
    return tag_to_idx, idx_to_tag


# -------------------------------------------------------------------------------------------#
# LOAD EMBEDDINGS
# -------------------------------------------------------------------------------------------#

def f_load_embeddings(filepath):
    """Load pre-trained word embeddings"""
    
    print(f"\nLoading embeddings from {filepath}...")
    
    data = torch.load(filepath)
    embeddings = data['embeddings']
    word_to_idx = data['word_to_index']
    
    embedding_dim = embeddings.shape[1]
    vocab_size = embeddings.shape[0]
    
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    
    return embeddings, word_to_idx, embedding_dim


# -------------------------------------------------------------------------------------------#
# PREPARE TRAINING DATA
# -------------------------------------------------------------------------------------------#

def f_get_word_embedding(word, embeddings, word_to_idx, embedding_dim, unk_vector=None):
    """Get embedding for a word, return zero vector if not in vocab"""
    
    word_lower = word.lower()
    
    if word_lower in word_to_idx:
        idx = word_to_idx[word_lower]
        return embeddings[idx]
    else:
        # Return small random vector for unknown words (not zeros)
        if unk_vector is None:
            # Use mean of all embeddings as UNK
            return torch.mean(embeddings, dim=0)
        else:
            return unk_vector

def f_create_context_vector(words, idx, context_window, embeddings, word_to_idx, embedding_dim, unk_vector):
    """Create context vector by concatenating embeddings of surrounding words"""
    
    context_vectors = []
    
    # Get left context
    for i in range(context_window, 0, -1):
        if idx - i >= 0:
            word = words[idx - i]
            vec = f_get_word_embedding(word, embeddings, word_to_idx, embedding_dim, unk_vector)
        else:
            # Padding for start of sentence
            vec = torch.zeros(embedding_dim)
        context_vectors.append(vec)
    
    # Get current word
    word = words[idx]
    vec = f_get_word_embedding(word, embeddings, word_to_idx, embedding_dim, unk_vector)
    context_vectors.append(vec)
    
    # Get right context
    for i in range(1, context_window + 1):
        if idx + i < len(words):
            word = words[idx + i]
            vec = f_get_word_embedding(word, embeddings, word_to_idx, embedding_dim, unk_vector)
        else:
            # Padding for end of sentence
            vec = torch.zeros(embedding_dim)
        context_vectors.append(vec)
    
    # Concatenate all context vectors
    context_vec = torch.cat(context_vectors)
    
    return context_vec


def f_prepare_dataset(tagged_sents, embeddings, word_to_idx, tag_to_idx, context_window, embedding_dim):
    """Prepare dataset for training"""
    
    X = []  # Context vectors
    y = []  # POS tags
    
    # Create UNK vector as mean of all embeddings
    unk_vector = torch.mean(embeddings, dim=0)


    for sent in tqdm(tagged_sents, desc="Preparing dataset"):
        words = [word for word, tag in sent]
        tags = [tag for word, tag in sent]
        
        for idx in range(len(words)):
            # Create context vector
            context_vec = f_create_context_vector(words, idx, context_window, embeddings, word_to_idx, embedding_dim, unk_vector)
            
            # Get tag index
            tag = tags[idx]
            tag_idx = tag_to_idx[tag]
            
            X.append(context_vec)
            y.append(tag_idx)
    
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    
    print(f"Dataset size: {len(X)} examples")
    print(f"Input shape: {X.shape}")
    
    return X, y


# -------------------------------------------------------------------------------------------#
# MLP MODEL
# -------------------------------------------------------------------------------------------#

class POSTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(POSTagger, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# -------------------------------------------------------------------------------------------#
# TRAINING
# -------------------------------------------------------------------------------------------#

def f_train_model(model, train_X, train_y, val_X, val_y, epochs, batch_size, learning_rate, device):
    """Train the POS tagger"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nTraining POS Tagger...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Create batches
        indices = torch.randperm(len(train_X))
        
        for i in tqdm(range(0, len(train_X), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_indices = indices[i:i+batch_size]
            batch_X = train_X[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_X_device = val_X.to(device)
            val_outputs = model(val_X_device)
            val_loss = criterion(val_outputs, val_y.to(device))
            
            _, predicted = torch.max(val_outputs, 1)
            val_accuracy = (predicted == val_y.to(device)).sum().item() / len(val_y)
        
        avg_train_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")
    
    print("Training complete!")
    return model


# -------------------------------------------------------------------------------------------#
# EVALUATION
# -------------------------------------------------------------------------------------------#

# def f_evaluate_model(model, test_X, test_y, idx_to_tag, device):
#     """Evaluate the model on test set"""
    
#     model.eval()
    
#     with torch.no_grad():
#         test_X_device = test_X.to(device)
#         outputs = model(test_X_device)
#         _, predicted = torch.max(outputs, 1)
        
#         accuracy = (predicted == test_y.to(device)).sum().item() / len(test_y)
    
#     print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
#     # Per-tag accuracy
#     tag_correct = {}
#     tag_total = {}
    
#     for true_idx, pred_idx in zip(test_y.numpy(), predicted.cpu().numpy()):
#         tag = idx_to_tag[true_idx]
        
#         if tag not in tag_total:
#             tag_total[tag] = 0
#             tag_correct[tag] = 0
        
#         tag_total[tag] += 1
#         if true_idx == pred_idx:
#             tag_correct[tag] += 1
    
#     print("\nPer-tag Accuracy:")
#     for tag in sorted(tag_total.keys()):
#         acc = tag_correct[tag] / tag_total[tag] * 100
#         print(f"  {tag:10s}: {acc:.2f}% ({tag_correct[tag]}/{tag_total[tag]})")
    
#     return accuracy




def f_evaluate_model(model, test_X, test_y, idx_to_tag, device):
    """Evaluate the model on test set"""
    
    model.eval()
    
    with torch.no_grad():
        test_X_device = test_X.to(device)
        outputs = model(test_X_device)
        _, predicted = torch.max(outputs, 1)
        
        accuracy = (predicted == test_y.to(device)).sum().item() / len(test_y)
    
    # Convert to numpy for sklearn metrics
    y_true = test_y.numpy()
    y_pred = predicted.cpu().numpy()
    
    # Calculate Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    
    # Per-tag metrics
    print("\nPer-tag Accuracy:")
    tag_correct = {}
    tag_total = {}
    
    for true_idx, pred_idx in zip(y_true, y_pred):
        tag = idx_to_tag[true_idx]
        
        if tag not in tag_total:
            tag_total[tag] = 0
            tag_correct[tag] = 0
        
        tag_total[tag] += 1
        if true_idx == pred_idx:
            tag_correct[tag] += 1
    
    for tag in sorted(tag_total.keys()):
        acc = tag_correct[tag] / tag_total[tag] * 100
        print(f"  {tag:10s}: {acc:.2f}% ({tag_correct[tag]}/{tag_total[tag]})")
    
    return accuracy, macro_f1, conf_matrix


def f_plot_confusion_matrix(conf_matrix, idx_to_tag, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    
    tags = [idx_to_tag[i] for i in range(len(idx_to_tag))]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=tags, yticklabels=tags)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - POS Tagging')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nConfusion matrix saved to {save_path}")





def f_load_glove_embeddings(filepath):
    """Load GloVe pre-trained embeddings"""
    
    print(f"\nLoading GloVe embeddings from {filepath}...")
    
    word_to_idx = {}
    embeddings_list = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 50000 == 0:
                print(f"  Loaded {idx} words...")
            
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            
            word_to_idx[word] = idx
            embeddings_list.append(vector)
    
    embeddings = torch.FloatTensor(np.array(embeddings_list))
    embedding_dim = embeddings.shape[1]
    
    print(f"  Loaded {len(word_to_idx)} words, embedding dim: {embedding_dim}")
    
    return embeddings, word_to_idx, embedding_dim




# -------------------------------------------------------------------------------------------#
# MAIN
# -------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data once
    tagged_sents = f_load_brown_corpus()
    train_data, val_data, test_data = f_split_data(tagged_sents)
    tag_to_idx, idx_to_tag = f_build_tag_vocab(tagged_sents)
    num_tags = len(tag_to_idx)
    
    # List of embeddings to test
    embedding_configs = [
        # ('embeddings/svd_woutsw.pt', 'SVD'),
        # ('embeddings/svd_wsw.pt', 'SVD'),
        ('embeddings/skipgram/skipgram_window_5_wsw.pt', 'Skip-Gram_wsw'),
        ('embeddings/cbow/cbow_window_5_wsw.pt', 'CBOW_wsw'),
        # ('glove_6B/glove.6B.100d.txt', 'GloVe'),  # Need special loader
    ]
    
    results = {}
    
    for emb_path, emb_name in embedding_configs:
        print("\n" + "="*80)
        print(f"TRAINING WITH {emb_name} EMBEDDINGS")
        print("="*80)
        
        try:
            # Load embeddings
            if emb_name == 'GloVe':
                embeddings, word_to_idx, embedding_dim = f_load_glove_embeddings(emb_path)
            else:
                embeddings, word_to_idx, embedding_dim = f_load_embeddings(emb_path)
            
            # Prepare datasets
            input_dim = embedding_dim * (2 * CONTEXT_WINDOW + 1)
            
            print("\nPreparing training data...")
            train_X, train_y = f_prepare_dataset(train_data, embeddings, word_to_idx, tag_to_idx, CONTEXT_WINDOW, embedding_dim)
            
            print("\nPreparing validation data...")
            val_X, val_y = f_prepare_dataset(val_data, embeddings, word_to_idx, tag_to_idx, CONTEXT_WINDOW, embedding_dim)
            
            print("\nPreparing test data...")
            test_X, test_y = f_prepare_dataset(test_data, embeddings, word_to_idx, tag_to_idx, CONTEXT_WINDOW, embedding_dim)
            
            # Create and train model
            model = POSTagger(input_dim, HIDDEN_DIM, num_tags, DROPOUT)
            model = f_train_model(model, train_X, train_y, val_X, val_y, EPOCHS, BATCH_SIZE, LEARNING_RATE, device)
            
            # Evaluate
            accuracy, macro_f1, conf_matrix = f_evaluate_model(model, test_X, test_y, idx_to_tag, device)
            
            results[emb_name] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'confusion_matrix': conf_matrix
            }
            
            # Save model
            import os
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'tag_to_idx': tag_to_idx,
                'idx_to_tag': idx_to_tag,
                'embedding_type': emb_name,
            }, f'models/pos_tagger_{emb_name.lower()}.pt')
            
            print(f"Model saved to models/pos_tagger_wsw_{emb_name.lower()}_E{EPOCHS}.pt")


        except FileNotFoundError:
            print(f"Skipping {emb_name}: File not found at {emb_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for emb_name, res in results.items():
        print(f"{emb_name:15s} - Accuracy: {res['accuracy']*100:.2f}%, Macro-F1: {res['macro_f1']:.4f}")