"""Centralized configuration."""

# Use bge-small-en: ~33MB, 512-token context, much better than MiniLM-L6
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Retrieval
TOP_K_DENSE = 2500
TOP_K_BM25 = 2500
TOP_K_FUSED = 3000          # after RRF
TOP_K_SUBMISSION = 100

# Experience band (JD: 5-9)
EXP_IDEAL_MIN = 5
EXP_IDEAL_MAX = 9

# Active candidate gates (hard)
LAST_ACTIVE_MAX_DAYS = 270   # ~9 months
MIN_RESPONSE_RATE = 0.05     # below this = clearly disengaged

# Service companies (JD explicit penalty)
SERVICE_COMPANIES = {
    "tcs", "tata consultancy services",
    "infosys", "infosys limited",
    "wipro", "wipro limited",
    "cognizant", "cognizant technology solutions",
    "capgemini",
    "accenture",
    "ibm global services", "ibm consulting",
    "deloitte consulting",
    "mindtree", "persistent",
    "ltimindtree", "hcl technologies", "hcl tech",
    "tech mahindra", "mphasis",
    "larsen & toubro infotech", "l&t infotech",
}

# JD core signals (tools + concepts)
CORE_TOOLS = [
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "sentence-transformers",
    "openai embeddings", "bge", "e5",
]

EVAL_KEYWORDS = [
    "ndcg", "mrr", "map", "mean average precision",
    "learning to rank", "learning-to-rank",
    "a/b test", "a/b testing", "ab test", "ab testing",
    "experimentation", "offline evaluation",
    "ranking quality", "retrieval quality",
    "evaluation framework",
]

RETRIEVAL_KEYWORDS = [
    "retrieval", "ranking", "relevance", "recommendation",
    "recommender", "matching", "search", "embedding",
    "hybrid search", "vector search", "semantic search",
    "ranking system", "search system",
    "personalization", "marketplace matching",
]

# Honeypot thresholds
HONEYPOT_SKILL_DURATION_MONTHS_MAX = 1   # expert with <1mo is sus
HONEYPOT_TITLE_EXP_YEARS_RATIO_MAX = 0.15  # senior+ with <15% expected yrs
