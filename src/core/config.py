"""
Configuration management for DataAlchemy
"""
import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"


class Config:
    """Centralized configuration for DataAlchemy"""
    
    # Model Configuration
    MODEL_PROVIDER = ModelProvider(os.getenv("MODEL_PROVIDER", "openai"))
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    
    # Performance Settings
    DEFAULT_PERFORMANCE_MODE = os.getenv("DEFAULT_PERFORMANCE_MODE", "medium")
    
    # Feature Engineering Settings
    MAX_POLYNOMIAL_DEGREE = int(os.getenv("MAX_POLYNOMIAL_DEGREE", "3"))
    MAX_INTERACTIONS = int(os.getenv("MAX_INTERACTIONS", "100"))
    MAX_CARDINALITY = int(os.getenv("MAX_CARDINALITY", "20"))
    MIN_FREQUENCY = int(os.getenv("MIN_FREQUENCY", "10"))
    
    # Data validation constants
    MIN_ROWS_REQUIRED = int(os.getenv("MIN_ROWS_REQUIRED", "100"))
    MIN_UNIQUE_VALUES = int(os.getenv("MIN_UNIQUE_VALUES", "2"))
    MAX_MEMORY_WARNING_MB = int(os.getenv("MAX_MEMORY_WARNING_MB", "500"))
    
    # Feature selection constants
    CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.95"))
    VARIANCE_THRESHOLD = float(os.getenv("VARIANCE_THRESHOLD", "0.01"))
    
    # Mathematical constants
    EPSILON = float(os.getenv("EPSILON", "1e-8"))
    MAX_EXP_INPUT = float(os.getenv("MAX_EXP_INPUT", "10.0"))
    
    # Temporal constants
    TEMPORAL_COMPONENTS = os.getenv("TEMPORAL_COMPONENTS", "year,month,day,dayofweek,is_weekend,month_sin,month_cos").split(',')
    
    # Logging
    ENABLE_PERFORMANCE_LOGGING = os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_model_string(cls) -> str:
        """Get the full model string for pydantic-ai"""
        provider = cls.MODEL_PROVIDER.value
        model = cls.MODEL_NAME
        
        # Special handling for different providers
        if provider == "openai":
            return f"openai:{model}"
        elif provider == "anthropic":
            return f"anthropic:{model}"
        elif provider == "gemini":
            return f"gemini:{model}"
        elif provider == "grok":
            return f"grok:{model}"
        else:
            raise ValueError(f"Unknown model provider: {provider}")
    
    @classmethod
    def get_transformer_configs(cls) -> Dict[str, Dict[str, Any]]:
        """Get configuration for transformers based on settings"""
        return {
            'polynomial': {
                'degrees': list(range(2, cls.MAX_POLYNOMIAL_DEGREE + 1)),
                'max_features': 50
            },
            'interaction': {
                'operations': ['multiply', 'divide'],
                'max_interactions': cls.MAX_INTERACTIONS
            },
            'temporal': {
                'components': [
                    'year', 'month', 'day', 'dayofweek', 'is_weekend',
                    'month_sin', 'month_cos'
                ]
            },
            'categorical': {
                'encoding_methods': ['one_hot', 'frequency'],
                'max_cardinality': cls.MAX_CARDINALITY,
                'min_frequency': cls.MIN_FREQUENCY
            },
            'mathematical': {
                'transforms': ['log', 'sqrt', 'reciprocal'],
                'create_binned': True
            }
        }
    
    @classmethod
    def validate(cls):
        """Validate configuration and raise errors if invalid"""
        # Check API key is set for the selected provider
        if cls.MODEL_PROVIDER == ModelProvider.OPENAI and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI provider")
        elif cls.MODEL_PROVIDER == ModelProvider.ANTHROPIC and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set when using Anthropic provider")
        elif cls.MODEL_PROVIDER == ModelProvider.GEMINI and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set when using Gemini provider")
        elif cls.MODEL_PROVIDER == ModelProvider.GROK and not cls.XAI_API_KEY:
            raise ValueError("XAI_API_KEY must be set when using Grok provider")
        
        # Validate model name matches provider
        valid_models = {
            ModelProvider.OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            ModelProvider.ANTHROPIC: [
                "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", 
                "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
            ],
            ModelProvider.GEMINI: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
            ModelProvider.GROK: ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it"]
        }
        
        if cls.MODEL_NAME not in valid_models.get(cls.MODEL_PROVIDER, []):
            print(f"Warning: Model {cls.MODEL_NAME} may not be valid for provider {cls.MODEL_PROVIDER.value}")
        
        return True


# Validate configuration on import
Config.validate()