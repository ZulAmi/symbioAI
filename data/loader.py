"""
Data Management System for Symbio AI

Handles data loading, preprocessing, validation, and caching
with support for multiple formats and distributed processing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata for datasets."""
    name: str
    format: str
    size: int
    schema: Dict[str, str]
    version: str
    created_at: str
    source: str
    preprocessing_applied: List[str]


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data and return transformed result."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data format and content."""
        pass


class JSONProcessor(DataProcessor):
    """Processor for JSON data."""
    
    async def process(self, data: Union[str, Path, Dict]) -> Dict[str, Any]:
        """Process JSON data."""
        if isinstance(data, (str, Path)):
            with open(data, 'r') as f:
                return json.load(f)
        return data
    
    def validate(self, data: Any) -> bool:
        """Validate JSON data."""
        try:
            if isinstance(data, (str, Path)):
                with open(data, 'r') as f:
                    json.load(f)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False


class CSVProcessor(DataProcessor):
    """Processor for CSV data."""
    
    def __init__(self, **pandas_kwargs):
        self.pandas_kwargs = pandas_kwargs
    
    async def process(self, data: Union[str, Path]) -> pd.DataFrame:
        """Process CSV data."""
        return pd.read_csv(data, **self.pandas_kwargs)
    
    def validate(self, data: Any) -> bool:
        """Validate CSV data."""
        try:
            pd.read_csv(data, nrows=1)
            return True
        except Exception:
            return False


class ParquetProcessor(DataProcessor):
    """Processor for Parquet data."""
    
    async def process(self, data: Union[str, Path]) -> pd.DataFrame:
        """Process Parquet data."""
        return pd.read_parquet(data)
    
    def validate(self, data: Any) -> bool:
        """Validate Parquet data."""
        try:
            pd.read_parquet(data, nrows=1) if hasattr(pd, 'read_parquet') else False
            return True
        except Exception:
            return False


class DataCache:
    """In-memory data cache with LRU eviction."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, data: Any) -> None:
        """Put data in cache."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            self.logger.debug(f"Evicted {lru_key} from cache")
        
        self._cache[key] = data
        self._access_order.append(key)
        self.logger.debug(f"Cached {key}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._access_order.clear()


class DataManager:
    """
    Central data management system for Symbio AI.
    
    Provides unified interface for loading, processing, and caching data
    from multiple sources and formats.
    """
    
    def __init__(self, config):
        self.config = config
        self.base_path = Path(config.base_path)
        self.cache = DataCache() if config.cache_enabled else None
        self.processors = self._initialize_processors()
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.logger = logging.getLogger(__name__)
    
    def _initialize_processors(self) -> Dict[str, DataProcessor]:
        """Initialize data processors for supported formats."""
        return {
            'json': JSONProcessor(),
            'csv': CSVProcessor(),
            'parquet': ParquetProcessor()
        }
    
    async def initialize(self) -> None:
        """Initialize the data management system."""
        self.logger.info("Initializing data management system")
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Scan for existing datasets
        await self._scan_datasets()
        
        self.logger.info(f"Data manager initialized with {len(self.datasets)} datasets")
    
    async def _scan_datasets(self) -> None:
        """Scan base path for existing datasets."""
        for format_ext in ['json', 'csv', 'parquet']:
            pattern = f"*.{format_ext}"
            for file_path in self.base_path.glob(pattern):
                await self._register_dataset(file_path)
    
    async def _register_dataset(self, file_path: Path) -> None:
        """Register a dataset in the system."""
        try:
            format_type = file_path.suffix[1:]  # Remove the dot
            if format_type in self.processors:
                metadata = DatasetMetadata(
                    name=file_path.stem,
                    format=format_type,
                    size=file_path.stat().st_size,
                    schema={},  # To be populated during first load
                    version="1.0.0",
                    created_at=str(file_path.stat().st_mtime),
                    source=str(file_path),
                    preprocessing_applied=[]
                )
                self.datasets[metadata.name] = metadata
                self.logger.debug(f"Registered dataset: {metadata.name}")
        except Exception as e:
            self.logger.warning(f"Failed to register dataset {file_path}: {e}")
    
    async def load_dataset(self, name: str, use_cache: bool = True) -> Any:
        """
        Load a dataset by name.
        
        Args:
            name: Dataset name
            use_cache: Whether to use cache if available
            
        Returns:
            Loaded dataset
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        # Check cache first
        if use_cache and self.cache:
            cached_data = self.cache.get(name)
            if cached_data is not None:
                self.logger.debug(f"Loaded {name} from cache")
                return cached_data
        
        # Load from source
        metadata = self.datasets[name]
        processor = self.processors[metadata.format]
        
        try:
            data = await processor.process(metadata.source)
            
            # Cache the data
            if use_cache and self.cache:
                self.cache.put(name, data)
            
            self.logger.info(f"Loaded dataset: {name}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {name}: {e}")
            raise
    
    async def add_dataset(self, name: str, data: Any, format_type: str) -> None:
        """
        Add a new dataset to the system.
        
        Args:
            name: Dataset name
            data: Dataset content
            format_type: Data format (json, csv, parquet)
        """
        if format_type not in self.processors:
            raise ValueError(f"Unsupported format: {format_type}")
        
        file_path = self.base_path / f"{name}.{format_type}"
        
        # Save data to file
        if format_type == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type == 'csv':
            data.to_csv(file_path, index=False)
        elif format_type == 'parquet':
            data.to_parquet(file_path)
        
        # Register the dataset
        await self._register_dataset(file_path)
        
        self.logger.info(f"Added dataset: {name}")
    
    async def preprocess_dataset(self, name: str, processors: List[str]) -> Any:
        """
        Apply preprocessing to a dataset.
        
        Args:
            name: Dataset name
            processors: List of preprocessing steps to apply
            
        Returns:
            Preprocessed data
        """
        data = await self.load_dataset(name)
        
        # Apply preprocessing steps
        for processor_name in processors:
            if hasattr(self, f'_preprocess_{processor_name}'):
                processor_func = getattr(self, f'_preprocess_{processor_name}')
                data = await processor_func(data)
            else:
                self.logger.warning(f"Unknown preprocessor: {processor_name}")
        
        # Update metadata
        if name in self.datasets:
            self.datasets[name].preprocessing_applied.extend(processors)
        
        return data
    
    async def _preprocess_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical columns."""
        numeric_columns = data.select_dtypes(include=['number']).columns
        data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
        return data
    
    async def _preprocess_remove_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove null values."""
        return data.dropna()
    
    async def _preprocess_encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = pd.Categorical(data[col]).codes
        return data
    
    def get_dataset_info(self, name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a dataset."""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets.keys())
    
    async def stream_dataset(self, name: str, batch_size: int = None) -> AsyncGenerator[Any, None]:
        """
        Stream dataset in batches for memory-efficient processing.
        
        Args:
            name: Dataset name
            batch_size: Size of each batch
            
        Yields:
            Data batches
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        metadata = self.datasets[name]
        
        if metadata.format == 'csv':
            # For CSV, we can use pandas chunking
            for chunk in pd.read_csv(metadata.source, chunksize=batch_size):
                yield chunk
        else:
            # For other formats, load full dataset and yield in batches
            data = await self.load_dataset(name)
            if isinstance(data, pd.DataFrame):
                for i in range(0, len(data), batch_size):
                    yield data.iloc[i:i + batch_size]
            elif isinstance(data, list):
                for i in range(0, len(data), batch_size):
                    yield data[i:i + batch_size]
            else:
                yield data
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.cache:
            self.cache.clear()
        self.logger.info("Data manager cleanup completed")