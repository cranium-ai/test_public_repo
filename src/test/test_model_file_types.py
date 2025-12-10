"""
Test file demonstrating various model types and model-like filenames
"""
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pickle
import json
import os

# Actual ML/AI models
def test_sentence_transformer():
    """Test using actual sentence transformer model"""
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    sentences = ['This is an example sentence', 'Each sentence is converted']
    embeddings = model.encode(sentences)
    return embeddings

def test_bertopic_model():
    """Test using BERTopic model"""
    topic_model = BERTopic()
    docs = ["This is a document about AI", "Machine learning is fascinating"]
    topics, probs = topic_model.fit_transform(docs)
    return topics, probs

# Filenames that could be interpreted as models
MODEL_FILES = [
    'my_claude_model.py',
    'gpt_model_v2.py',
    'custom_llm_model.py',
    '../apis/accessanalyzer-2019-11-01.min.json',
    './models/trained_classifier.pkl',
    'bert_finetuned_model.pt',
    'neural_network_model.h5',
    'transformers_model.bin',
    'sklearn_pipeline_model.joblib'
]

def test_model_file_paths():
    """Test various model-like file paths"""
    model_paths = {
        'claude_api': 'my_claude_model.py',
        'aws_api': '../apis/accessanalyzer-2019-11-01.min.json',
        'pickle_model': './models/trained_classifier.pkl',
        'pytorch_model': 'bert_finetuned_model.pt',
        'keras_model': 'neural_network_model.h5'
    }

    for name, path in model_paths.items():
        print(f"Model type: {name}, Path: {path}")

    return model_paths

def save_dummy_model_files():
    """Create dummy model files for testing"""
    # Save a simple pickle model
    dummy_model = {'weights': [0.1, 0.2, 0.3], 'bias': 0.5}
    with open('trained_classifier.pkl', 'wb') as f:
        pickle.dump(dummy_model, f)

    # Save a JSON config that looks like a model
    api_config = {
        'version': '2019-11-01',
        'metadata': {
            'apiVersion': '2019-11-01',
            'endpointPrefix': 'access-analyzer'
        }
    }
    with open('accessanalyzer-2019-11-01.min.json', 'w') as f:
        json.dump(api_config, f)

def load_model_from_file(filename):
    """Load a model from various file types"""
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.py'):
        # Reference to a Python model file
        return f"Python model module: {filename}"
    else:
        return f"Unknown model format: {filename}"

if __name__ == '__main__':
    print("Testing actual ML models:")
    print("=" * 50)

    # Test sentence transformer
    print("\n1. Testing distilbert-base-nli-mean-tokens:")
    embeddings = test_sentence_transformer()
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Test BERTopic
    print("\n2. Testing BERTopic:")
    topics, probs = test_bertopic_model()
    print(f"Topics: {topics}")

    print("\n\nTesting model-like filenames:")
    print("=" * 50)

    # Test model file paths
    print("\n3. Model file paths:")
    model_paths = test_model_file_paths()

    print("\n4. All model files to check:")
    for model_file in MODEL_FILES:
        print(f"  - {model_file}")
