import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os

# hyperparameters - improved settings
batch_size = 64
block_size = 256
max_iters = 8000  # Increased for better training
eval_interval = 400
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Style parameters
n_styles = 2  # Twain=0, Dickens=1
classification_block_size = 256  # Match generation block size

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # Dynamically create the causal mask
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class StyleClassifier(nn.Module):
    """Classifier to distinguish Twain vs Dickens"""
    
    def __init__(self, vocab_size, n_embd, n_styles):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(classification_block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Classification head
        self.classifier_head = nn.Linear(n_embd, n_styles)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, idx):
        B, T = idx.shape
        T = min(T, classification_block_size)
        idx = idx[:, :T]
        
        tok_emb = self.token_embedding_table(idx)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, n_embd)
        x = self.dropout(x)
        
        logits = self.classifier_head(x)  # (B, n_styles)
        return logits

class StyleConditionalGenerator(nn.Module):
    """Generator with style conditioning"""
    
    def __init__(self, vocab_size, n_embd, n_styles):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Style embedding
        self.style_embedding_table = nn.Embedding(n_styles, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Better weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, style_id, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        
        # Add style embedding (broadcast across all positions)
        style_emb = self.style_embedding_table(style_id).unsqueeze(1)  # (B,1,C)
        style_emb = style_emb.expand(-1, T, -1)  # (B,T,C)
        
        x = tok_emb + pos_emb + style_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, style_id, max_new_tokens):
        """Generate text in specified style"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond, style_id)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def clean_text(text):
    """Clean text while preserving structure"""
    # Remove excessive whitespace but keep paragraph structure
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1]:  # Keep paragraph breaks
            cleaned_lines.append('')
    
    return '\n'.join(cleaned_lines)

def balance_texts(twain_text, dickens_text):
    """Balance text lengths to prevent bias"""
    twain_len = len(twain_text)
    dickens_len = len(dickens_text)
    
    print(f"Original lengths - Twain: {twain_len}, Dickens: {dickens_len}")
    
    # Use the shorter length for both
    min_len = min(twain_len, dickens_len)
    
    if twain_len > min_len:
        twain_text = twain_text[:min_len]
        print(f"Truncated Twain to {min_len} characters")
    
    if dickens_len > min_len:
        dickens_text = dickens_text[:min_len]
        print(f"Truncated Dickens to {min_len} characters")
    
    return twain_text, dickens_text

def prepare_classification_data(twain_text, dickens_text, stoi):
    """Prepare classification training data"""
    def create_chunks(text, author_id, chunk_size=classification_block_size):
        chunks = []
        
        # Split by sentences for better chunks
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if len(current_chunk.strip()) > 50:  # Minimum chunk size
                    encoded = [stoi.get(c, 0) for c in current_chunk[:chunk_size]]
                    
                    # Pad if necessary
                    if len(encoded) < chunk_size:
                        encoded += [0] * (chunk_size - len(encoded))
                    
                    chunks.append((torch.tensor(encoded, dtype=torch.long), author_id))
                current_chunk = sentence + ". "
        
        # Don't forget the last chunk
        if len(current_chunk.strip()) > 50:
            encoded = [stoi.get(c, 0) for c in current_chunk[:chunk_size]]
            if len(encoded) < chunk_size:
                encoded += [0] * (chunk_size - len(encoded))
            chunks.append((torch.tensor(encoded, dtype=torch.long), author_id))
        
        return chunks
    
    twain_chunks = create_chunks(twain_text, 0)  # Twain = 0
    dickens_chunks = create_chunks(dickens_text, 1)  # Dickens = 1
    
    classification_data = twain_chunks + dickens_chunks
    
    print(f"Classification data - Twain chunks: {len(twain_chunks)}, Dickens chunks: {len(dickens_chunks)}")
    
    return classification_data

def prepare_generation_data(twain_text, dickens_text, stoi):
    """FIXED: Properly track which chunks belong to which author"""
    generation_data = []
    
    # Process Twain chunks with correct labels
    twain_encoded = [stoi.get(c, 0) for c in twain_text]
    twain_count = 0
    for i in range(0, len(twain_encoded) - block_size, block_size//2):
        chunk = twain_encoded[i:i+block_size+1]
        if len(chunk) == block_size + 1:
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            style = 0  # Twain
            generation_data.append((x, y, style))
            twain_count += 1
    
    # Process Dickens chunks with correct labels  
    dickens_encoded = [stoi.get(c, 0) for c in dickens_text]
    dickens_count = 0
    for i in range(0, len(dickens_encoded) - block_size, block_size//2):
        chunk = dickens_encoded[i:i+block_size+1]
        if len(chunk) == block_size + 1:
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            style = 1  # Dickens
            generation_data.append((x, y, style))
            dickens_count += 1
    
    print(f"Generation data - Twain chunks: {twain_count}, Dickens chunks: {dickens_count}")
    
    return generation_data

def get_classification_batch(classification_data, batch_size):
    """Get batch for classification training"""
    indices = torch.randint(len(classification_data), (batch_size,))
    
    texts = torch.stack([classification_data[i][0] for i in indices])
    labels = torch.tensor([classification_data[i][1] for i in indices], dtype=torch.long)
    
    return texts.to(device), labels.to(device)

def get_generation_batch(generation_data, batch_size):
    """FIXED: Use the correct style labels for each chunk"""
    indices = torch.randint(len(generation_data), (batch_size,))
    
    x = torch.stack([generation_data[i][0] for i in indices])
    y = torch.stack([generation_data[i][1] for i in indices])
    styles = torch.tensor([generation_data[i][2] for i in indices], dtype=torch.long)
    
    return x.to(device), y.to(device), styles.to(device)

# def train_models(classification_data, generation_data, vocab_size):
def train_models(classification_data, generation_data, vocab_size, stoi, itos):

    """Train both classifier and generator with improvements"""
    
    # Initialize models
    classifier = StyleClassifier(vocab_size, n_embd, n_styles).to(device)
    generator = StyleConditionalGenerator(vocab_size, n_embd, n_styles).to(device)
    
    # Separate learning rates - generation is harder
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate * 0.3)  # Lower for generation
    
    # Learning rate schedulers
    classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, max_iters)
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, max_iters)
    
    # Track best models
    best_accuracy = 0.0
    best_gen_loss = float('inf')
    
    for iter in range(max_iters):
        
        # Train classifier
        classifier.train()
        texts, labels = get_classification_batch(classification_data, batch_size//2)
        
        logits = classifier(texts)
        classification_loss = F.cross_entropy(logits, labels)
        
        classifier_optimizer.zero_grad()
        classification_loss.backward()
        classifier_optimizer.step()
        classifier_scheduler.step()
        
        # Train generator
        generator.train()
        x, y, styles = get_generation_batch(generation_data, batch_size//2)
        
        logits, generation_loss = generator(x, styles, y)
        
        generator_optimizer.zero_grad()
        generation_loss.backward()
        generator_optimizer.step()
        generator_scheduler.step()
        
        if iter % eval_interval == 0:
            # Evaluate classifier
            classifier.eval()
            generator.eval()
            
            with torch.no_grad():
                # Classification accuracy
                test_texts, test_labels = get_classification_batch(classification_data, 200)
                test_logits = classifier(test_texts)
                predictions = torch.argmax(test_logits, dim=-1)
                accuracy = (predictions == test_labels).float().mean()
                
                # Generation evaluation
                eval_x, eval_y, eval_styles = get_generation_batch(generation_data, 100)
                _, eval_gen_loss = generator(eval_x, eval_styles, eval_y)
            
            print(f"Step {iter}: Class Loss: {classification_loss:.4f}, Gen Loss: {generation_loss:.4f}, "
                  f"Accuracy: {accuracy:.3f}, Eval Gen Loss: {eval_gen_loss:.4f}")
            
            # Save best models
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"New best accuracy: {best_accuracy:.3f}")
            
            if eval_gen_loss < best_gen_loss:
                best_gen_loss = eval_gen_loss
                torch.save({
                    'classifier_state_dict': classifier.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'vocab_size': vocab_size,
                    'stoi': stoi,
                    'itos': itos,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'n_styles': n_styles,
                    'best_gen_loss': best_gen_loss,
                    'best_accuracy': best_accuracy
                }, 'best_twain_dickens_models.pth')
                print(f"Saved best model! Gen loss: {best_gen_loss:.4f}")
            
            # Test generation quality
            if iter % (eval_interval * 2) == 0:
                sample_twain = generate_text_in_style(generator, "Mark Twain", 80, stoi, itos)
                sample_dickens = generate_text_in_style(generator, "Charles Dickens", 80, stoi, itos)
                print(f"\nTwain sample: {sample_twain[:100]}...")
                print(f"Dickens sample: {sample_dickens[:100]}...\n")
    
    return classifier, generator

def classify_text(text, classifier, stoi, itos):
    """Classify a text sample as Twain or Dickens"""
    # Encode text
    encoded = [stoi.get(c, 0) for c in text[:classification_block_size]]
    if len(encoded) < classification_block_size:
        encoded += [0] * (classification_block_size - len(encoded))
    
    # input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    classifier = classifier.to(input_tensor.device)

    
    classifier.eval()
    with torch.no_grad():
        logits = classifier(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
    
    authors = ["Mark Twain", "Charles Dickens"]
    confidence = probabilities[0][prediction].item()
    
    return {
        "predicted_author": authors[prediction],
        "confidence": confidence,
        "probabilities": {
            "Mark Twain": probabilities[0][0].item(),
            "Charles Dickens": probabilities[0][1].item()
        }
    }

def generate_text_in_style(generator, author_choice, num_tokens, stoi, itos):
    """Generate text in chosen author's style with better prompting"""
    # Better starting prompts
    if author_choice == "Mark Twain":
        prompt = "I was "
    else:
        prompt = "It was "
    
    # # Encode prompt
    # context = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    
    # # Convert author choice to style_id
    # style_id = torch.tensor([0 if author_choice == "Mark Twain" else 1], device=device)

    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    style_id = torch.tensor([0 if author_choice == "Mark Twain" else 1])

    context = context.to(generator.lm_head.weight.device)
    style_id = style_id.to(generator.lm_head.weight.device)

    
    generator.eval()
    with torch.no_grad():
        generated = generator.generate(context, style_id, num_tokens)
        text = ''.join([itos.get(i, '') for i in generated[0].tolist()])
    
    return text

def main():
    print("=== Twain vs Dickens Style Analyzer ===")
    
    # Load data
    print("Loading Twain data...")
    with open('twain.txt', 'r', encoding='utf-8') as f:
        twain_text = f.read()
    
    print("Loading Dickens data...")
    with open('dickens.txt', 'r', encoding='utf-8') as f:
        dickens_text = f.read()
    
    # Clean and balance texts
    print("Cleaning texts...")
    twain_text = clean_text(twain_text)
    dickens_text = clean_text(dickens_text)
    
    print("Balancing text lengths...")
    twain_text, dickens_text = balance_texts(twain_text, dickens_text)
    
    # Create vocabulary
    print("Creating vocabulary...")
    all_text = twain_text + dickens_text
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Prepare datasets
    print("Preparing datasets...")
    classification_data = prepare_classification_data(twain_text, dickens_text, stoi)
    generation_data = prepare_generation_data(twain_text, dickens_text, stoi)
    
    print(f"Total classification samples: {len(classification_data)}")
    print(f"Total generation samples: {len(generation_data)}")
    
    # Train models
    print("Starting training...")
    # classifier, generator = train_models(classification_data, generation_data, vocab_size)
    classifier, generator = train_models(classification_data, generation_data, vocab_size, stoi, itos)

    
    # Save final models
    print("Saving final models...")
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'n_styles': n_styles
    }, 'final_twain_dickens_models.pth')
    
    print("Training complete!")
    
    # Quick tests
    print("\n=== Quick Tests ===")
    
    # Test with samples that should be in training data
    twain_sample = "My father was a St. Bernard, my mother was a collie"
    dickens_sample = "It was the best of times, it was the worst of times"
    
    result1 = classify_text(twain_sample, classifier, stoi, itos)
    print(f"Twain sample: '{twain_sample}'")
    print(f"Predicted: {result1['predicted_author']} (confidence: {result1['confidence']:.3f})")
    
    result2 = classify_text(dickens_sample, classifier, stoi, itos)
    print(f"\nDickens sample: '{dickens_sample}'")
    print(f"Predicted: {result2['predicted_author']} (confidence: {result2['confidence']:.3f})")
    
    # Generate samples
    print("\n=== Generated Samples ===")
    twain_gen = generate_text_in_style(generator, "Mark Twain", 150, stoi, itos)
    dickens_gen = generate_text_in_style(generator, "Charles Dickens", 150, stoi, itos)
    
    print(f"Generated Twain-style text:\n{twain_gen}\n")
    print(f"Generated Dickens-style text:\n{dickens_gen}")

if __name__ == "__main__":
    main()