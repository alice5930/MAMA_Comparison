# Multi-Agent Model AI System (MAMA)
The MAMA framework processes 336,776 real flight records using trust-based agent selection and semantic similarity matching.

### Core Components

- **Verifiable Reputation Ledger (VRL)**: Hash-chained trust records with tamper-resistant audit trail
- **SBERT Engine**: Semantic similarity computation using all-MiniLM-L6-v2 transformer model
- **MARL System**: Multi-Agent Reinforcement Learning with Q-Learning for dynamic agent selection
- **LTR Engine**: Learning-to-Rank neural network for flight recommendation optimization
- **Multi-Agent Collaboration**: 5 specialized agents (Weather, Safety, Flight Info, Economic, Integration)

## Installation
```bash
pip install -r requirements.txt
```

### Single Flight Query
```bash
python main.py --departure "New York" --destination "Los Angeles" --date "2024-12-15"
```

### Interactive Mode
```bash
python main.py --interactive
```

### Basic Functionality Tests
```bash
# Test examples
python main.py --departure "Chicago" --destination "Miami" --date "2024-12-20"
python main.py --departure "Boston" --destination "San Francisco" --date "2024-11-15"
```

# Run robustness and scalability tests
python main.py --experiments
```

## Data Sources

- **flights.csv**: 336,776 real flight records from US domestic flights dataset
- **test_queries_150.json**: 150 standardized test queries for evaluation
- **vrl_records.json**: Trust records with hash chain integrity verification

## Evaluation

The system includes the following evaluation metrics:
- **MRR (Mean Reciprocal Rank)**: Ranking quality measurement
- **NDCG@5**: Normalized Discounted Cumulative Gain at top 5 positions
- **ART (Average Response Time)**: System efficiency measurement

## Workflow

### Phase 1: Semantic Query Analysis
- SBERT-based query embedding and agent similarity computation
- Agent selection using SelectionScore = α·SBERT_similarity + (1-α)·TrustScore

### Phase 2: Multi-Agent Coordination
- Parallel execution of specialized agents on real flight data
- Trust-weighted result aggregation and conflict resolution

### Phase 3: Decision Integration
- LTR-based flight ranking with learned preferences
- Multi-criteria decision analysis (MCDA) integration

### Phase 4: Trust Evolution
- VRL updates with performance evidence
- Hash chain integrity maintenance and verification