## Chapter 2: Tokens and Embeddings

### **Overview**

Chapter 2 delves into how text is transformed into a format that large language models (LLMs) can process, focusing on tokenization and embeddings. These are foundational concepts for understanding how LLMs "see" and work with language.

---

### **1. LLM Tokenization**

- **Tokenization** is the process of breaking down raw text into smaller units called *tokens*.
- Tokens can be words, subwords, characters, or bytes, depending on the tokenizer design.
- LLMs do not process raw text directly; they require input to be in the form of tokens, each mapped to a unique integer ID.

---

### **2. How Tokenizers Prepare Inputs**

- Tokenizers convert text into sequences of tokens, which are then mapped to integer IDs using a vocabulary.
- Example: The sentence "Language models are powerful" might be tokenized as ["Language", "models", "are", "power", "ful"] and mapped to IDs like [502, 1037, 312, 987, 421].
- Special tokens are often added for tasks (e.g., `[CLS]`, `[SEP]` in BERT).

---

### **3. Downloading and Running an LLM**

- Pretrained LLMs and their tokenizers can be downloaded using libraries like Hugging Face Transformers.
- The typical workflow involves:
  - Loading the tokenizer and model.
  - Tokenizing input text.
  - Passing token IDs to the model for inference.

---

### **4. How Does the Tokenizer Break Down Text?**

- Tokenizers use algorithms (e.g., Byte Pair Encoding, WordPiece, SentencePiece) to split text into tokens.
- The goal is to balance vocabulary size and the ability to represent rare or unseen words.
- Example: "unhappiness" might be split into ["un", "happiness"] or ["un", "happy", "ness"].

---

### **5. Word vs. Subword vs. Character vs. Byte Tokens**

- **Word tokens:** Each word is a token (simple, but large vocabulary and poor handling of rare words).
- **Subword tokens:** Words are split into frequent subword units (balances vocabulary size and coverage).
- **Character tokens:** Each character is a token (small vocabulary, but long sequences).
- **Byte tokens:** Each byte is a token (used in multilingual or byte-level models).

| Tokenization Type | Pros | Cons |
|-------------------|------|------|
| Word              | Simple, intuitive | Huge vocab, can't handle rare words |
| Subword           | Handles rare words, compact vocab | Slightly less intuitive |
| Character         | Small vocab, language-agnostic | Long sequences, less efficient |
| Byte              | Universal, handles any text | Longest sequences |

---

### **6. Comparing Trained LLM Tokenizers**

- Different LLMs use different tokenization strategies, affecting how they split and represent text.
- Tokenizer choice impacts model efficiency, performance on rare words, and multilingual capabilities.
- Examples:
  - GPT-2 uses byte-level BPE.
  - BERT uses WordPiece.
  - XLM-R uses SentencePiece.

---

### **7. Tokenizer Properties**

- **Vocabulary size:** Number of unique tokens the model can recognize.
- **Coverage:** Ability to represent all possible input texts.
- **Efficiency:** Speed and memory usage during tokenization and detokenization.
- **Language support:** Some tokenizers are optimized for specific languages or scripts.

---

### **8. Token Embeddings**

- Each token ID is mapped to a high-dimensional vector (embedding) in the model.
- The embedding matrix is learned during pretraining.
- Embeddings capture semantic and syntactic properties of tokens.

---

### **9. A Language Model Holds Embeddings for the Vocabulary of Its Tokenizer**

- The embedding layer is the first layer of most LLMs.
- Each token in the vocabulary has a corresponding embedding vector.
- The model uses these vectors as the input for further processing.

---

### **10. Creating Contextualized Word Embeddings with Language Models**

- LLMs generate *contextualized embeddings*: the representation of a word depends on its context in the sentence.
- Example: The embedding for "bank" in "river bank" differs from "bank" in "bank account".
- Contextualization is achieved through the model's deep architecture and attention mechanisms.

---

### **11. Text Embeddings (for Sentences and Whole Documents)**

- LLMs can generate embeddings for entire sentences, paragraphs, or documents.
- These embeddings are useful for tasks like semantic search, clustering, and classification.
- Pooling strategies (e.g., mean pooling, using `[CLS]` token) are used to aggregate token embeddings.

---

### **12. Word Embeddings Beyond LLMs**

- Before LLMs, standalone word embeddings (Word2Vec, GloVe, FastText) were widely used.
- These provide static embeddings (same vector for a word regardless of context).
- Still useful for some applications due to simplicity and efficiency.

---

### **13. Using Pretrained Word Embeddings**

- Pretrained embeddings can be loaded and used in downstream models.
- They provide a strong initialization, especially when labeled data is limited.
- Common sources: GloVe, Word2Vec, FastText.

---

### **14. The Word2vec Algorithm and Contrastive Training**

- **Word2Vec:** Trains embeddings by predicting a word from its context (CBOW) or context from a word (Skip-gram).
- **Contrastive training:** Encourages similar words to have similar embeddings and dissimilar words to be far apart.
- These methods laid the groundwork for modern embedding techniques.

---

### **15. Embeddings for Recommendation Systems**

- Embeddings can represent not just words, but also items like songs, products, or users.
- Similarity in embedding space can be used for recommendations (e.g., recommend songs similar to those a user likes).

---

### **16. Recommending Songs by Embeddings**

- Songs are mapped to embedding vectors based on metadata, lyrics, or user interaction data.
- User preferences are also embedded, and recommendations are made by finding songs close to the user's embedding.

---

### **17. Training a Song Embedding Model**

- Models are trained to place similar songs (or songs liked by similar users) close together in embedding space.
- Contrastive or triplet loss functions are commonly used.

---

### **18. Summary**

- Tokenization and embeddings are the foundation of how LLMs process language.
- Tokenizers break text into manageable units; embeddings map these units into vectors the model can understand.
- Contextual embeddings from LLMs enable nuanced understanding and powerful applications across NLP and beyond.
