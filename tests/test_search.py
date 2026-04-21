from src.config import Config
import pytest

def test_config_loads():
    """Check if config class exists and attributes are reachable"""
    assert hasattr(Config, 'AZURE_OPENAI_API_KEY')

def test_imports():
    """Test that main modules can be imported"""
    from src.search_engine import RibbonSearchEngine
    from src.data_loader import RibbonDataLoader
    assert RibbonSearchEngine is not None
    assert RibbonDataLoader is not None