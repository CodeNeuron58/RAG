## 🧩 Architecture Summary

The RAG pipeline consists of the following key components:

| Step | Component                    | Description                                                                           |
| ---- | ---------------------------- | ------------------------------------------------------------------------------------- |
| 1    | **Document Loading**         | Load PDF documents and extract text.                                                  |
| 2    | **Text Chunking**            | Split long text into smaller, overlapping pieces for better retrieval.                |
| 3    | **Embedding Generation**     | Convert text chunks into numerical vectors using a HuggingFace embedding model.       |
| 4    | **Vector Storage**           | Store these embeddings in a vector database (Chroma) for efficient similarity search. |
| 5    | **Keyword Retrieval (BM25)** | Perform keyword-based search using classical BM25 algorithm.                          |
| 6    | **Hybrid Retriever**         | Combine both semantic and keyword search for better results.                          |
| 7    | **Language Model (LLM)**     | Use a quantized large language model to generate natural language answers.            |
| 8    | **Retrieval-QA Chain**       | Integrate retriever + LLM to form a complete RAG system.                              |

---

## 📄 Step-by-Step Explanation

### 1️⃣ Document Loading

The pipeline starts by loading a **PDF document** using `PyPDFLoader`.
This loader extracts text from the file and prepares it for processing.

---

### 2️⃣ Text Splitting

Long texts are broken into **smaller chunks** using `RecursiveCharacterTextSplitter`.
This ensures that:

* Each chunk is short enough for the model to handle,
* Overlapping text preserves context between consecutive chunks.

For example, if a PDF section has 1000 characters, it can be split into overlapping chunks of 200 characters each.

This improves both **retrieval accuracy** and **answer completeness**.

---

### 3️⃣ Embedding Generation

Each text chunk is converted into a **dense vector representation** using the model
`BAAI/bge-base-en-v1.5`.

Embeddings capture **semantic meaning** — words and sentences that mean similar things have vectors that are close together in the embedding space.

These embeddings are generated through the **HuggingFace Inference API**, using your API token.

---

### 4️⃣ Vector Storage (Chroma)

The generated embeddings are stored in a **Chroma** vector database.
When a user asks a question, Chroma computes **cosine similarity** between the query embedding and stored document embeddings to find the most relevant chunks.

This forms the **dense retriever**, also known as the **vector retriever**.

---

### 5️⃣ Sparse Retrieval (BM25)

In parallel, a **BM25 retriever** is created using the same text chunks.

BM25 is a **keyword-based** retrieval algorithm that scores documents based on how many query words appear in them, adjusted by term frequency and document length.

* It’s called **sparse** because it uses a sparse representation (each word is a dimension).
* It doesn’t understand meaning — it only matches exact words.

---

### 6️⃣ Dense vs. Sparse Search

**Dense Search (Vector-Based / Semantic):**

* Represents text as continuous numerical vectors (embeddings).
* Finds semantically similar documents (e.g., “kitten” ≈ “cat”).
* Great at understanding meaning, synonyms, and paraphrases.
* Weak at exact matching for domain-specific terms.

**Sparse Search (BM25 / Keyword-Based):**

* Represents text as bag-of-words.
* Finds documents with overlapping keywords (e.g., “cat” ≠ “kitten”).
* Excellent for exact matches and precise term retrieval.
* Weak at semantic understanding.

---

### 7️⃣ Hybrid Retriever

To get the best of both worlds, we combine **dense** and **sparse** retrievers into an **Ensemble Retriever**.

It calculates a **hybrid score** as:

[
\text{hybrid_score} = (1 - \alpha) \times \text{sparse_score} + \alpha \times \text{dense_score}
]

This way:

* Exact keyword matches get rewarded via BM25.
* Semantic understanding is preserved via embeddings.

The result is a **robust hybrid retriever** that performs well on both literal and conceptual queries.

---

### 8️⃣ Language Model (LLM) Setup

The text generation model used is **HuggingFaceH4/zephyr-7b-beta**, loaded with **4-bit quantization** to optimize memory and performance.

Quantization reduces model size while maintaining good accuracy, allowing large models to run on smaller GPUs.

The model and tokenizer are loaded using the HuggingFace `transformers` library with `bitsandbytes` for 4-bit support.

---

### 9️⃣ Text Generation Pipeline

A **text-generation pipeline** is created using HuggingFace, and wrapped with LangChain’s `HuggingFacePipeline` class to integrate with other LangChain components.

This LLM will take retrieved context from the retriever and generate final, natural-language answers.

---

### 🔟 Retrieval-QA Chains

Two separate RAG chains are created using `RetrievalQA`:

1. **Normal Chain:** Uses only the dense vector retriever (semantic search).
2. **Hybrid Chain:** Uses the hybrid retriever (semantic + keyword search).

When a user asks a question (e.g., *“What is Abstractive Question Answering?”*), the chain:

1. Retrieves the top relevant document chunks.
2. Feeds them to the LLM.
3. Generates a grounded answer.

---

## 📊 Output Comparison

When you run both chains, you’ll typically find:

* The **normal (vector-only)** chain understands meaning well but might miss exact keywords.
* The **hybrid** chain balances both and gives more complete, accurate, and contextually relevant answers.

---

## ⚙️ Tools and Libraries Used

| Library                       | Purpose                                               |
| ----------------------------- | ----------------------------------------------------- |
| **LangChain**                 | Orchestration of retrievers, chains, and pipelines.   |
| **PyPDFLoader**               | Extracts text from PDF files.                         |
| **Chroma**                    | Lightweight vector database for semantic search.      |
| **HuggingFace Transformers**  | Loads the Zephyr-7B model for text generation.        |
| **BitsAndBytes**              | Enables efficient 4-bit quantization of large models. |
| **Rank-BM25**                 | Implements keyword-based retrieval.                   |
| **HuggingFace Inference API** | Generates embeddings for document chunks.             |

---

## 💡 Why Hybrid Search Matters

| Problem                           | Solution                                   |
| --------------------------------- | ------------------------------------------ |
| Exact keyword queries             | BM25 (sparse) ensures accurate match       |
| Paraphrased or conceptual queries | Dense embeddings capture meaning           |
| Domain-specific terms or formulas | Sparse search handles technical vocabulary |
| Contextual understanding          | Dense search understands semantics         |
| Best of both worlds               | Hybrid search combines both strengths      |

---

## 📚 Key Takeaways

* **Dense retrievers** understand meaning but not exact words.
* **Sparse retrievers** match words exactly but miss semantic connections.
* **Hybrid retrievers** balance both — ideal for real-world search tasks.
* **RAG** grounds LLM responses in factual, document-based context — reducing hallucinations.
* **Quantization** allows running large models efficiently on limited hardware.