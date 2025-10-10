"""
Database models and data persistence layer for Symbio AI.

Provides ORM models, database migrations, connection pooling,
and data access patterns for production environments.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone
import uuid
import hashlib
from pathlib import Path
import asyncpg
import sqlite3
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import select, insert, update, delete
from alembic import command
from alembic.config import Config


class DatabaseType(Enum):
    """Database type enumeration."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"


class ModelStatus(Enum):
    """Model status enumeration."""
    ACTIVE = "active"
    TRAINING = "training"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = "symbio_ai"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    migration_path: str = "alembic"
    
    @property
    def connection_url(self) -> str:
        """Get database connection URL."""
        if self.database_type == DatabaseType.SQLITE:
            return f"sqlite+aiosqlite:///{self.database}.db"
        elif self.database_type == DatabaseType.POSTGRESQL:
            return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.database_type == DatabaseType.MYSQL:
            return f"mysql+aiomysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


# SQLAlchemy Base
Base = declarative_base()


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(255))
    last_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    api_key = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="user")
    training_jobs = relationship("TrainingJob", back_populates="user")
    models = relationship("Model", back_populates="user")


class Model(Base):
    """AI Model model."""
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)
    architecture = Column(Text)
    description = Column(Text)
    status = Column(String(50), default=ModelStatus.ACTIVE.value)
    file_path = Column(String(500))
    file_size = Column(Integer)
    checksum = Column(String(64))
    metadata = Column(JSON)
    parameters = Column(JSON)
    performance_metrics = Column(JSON)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="models")
    training_jobs = relationship("TrainingJob", back_populates="model")
    inference_jobs = relationship("InferenceJob", back_populates="model")


class Dataset(Base):
    """Dataset model."""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    data_type = Column(String(100), nullable=False)
    format = Column(String(50))
    file_path = Column(String(500))
    file_size = Column(Integer)
    record_count = Column(Integer)
    checksum = Column(String(64))
    schema = Column(JSON)
    metadata = Column(JSON)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="dataset")


class Experiment(Base):
    """Experiment model."""
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default=ExperimentStatus.DRAFT.value)
    configuration = Column(JSON)
    results = Column(JSON)
    metrics = Column(JSON)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="experiments")
    training_jobs = relationship("TrainingJob", back_populates="experiment")


class TrainingJob(Base):
    """Training job model."""
    __tablename__ = "training_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    status = Column(String(50), default=TrainingStatus.PENDING.value)
    model_id = Column(String, ForeignKey("models.id"))
    dataset_id = Column(String, ForeignKey("datasets.id"))
    experiment_id = Column(String, ForeignKey("experiments.id"))
    user_id = Column(String, ForeignKey("users.id"))
    configuration = Column(JSON)
    hyperparameters = Column(JSON)
    metrics = Column(JSON)
    logs = Column(Text)
    error_message = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    estimated_duration = Column(Integer)  # seconds
    progress = Column(Float, default=0.0)
    resource_usage = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    model = relationship("Model", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    experiment = relationship("Experiment", back_populates="training_jobs")
    user = relationship("User", back_populates="training_jobs")


class InferenceJob(Base):
    """Inference job model."""
    __tablename__ = "inference_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, ForeignKey("models.id"))
    input_data = Column(JSON)
    output_data = Column(JSON)
    batch_size = Column(Integer, default=1)
    latency_ms = Column(Float)
    memory_usage_mb = Column(Float)
    status = Column(String(50))
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    model = relationship("Model", back_populates="inference_jobs")


class SystemLog(Base):
    """System log model."""
    __tablename__ = "system_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    module = Column(String(255))
    function = Column(String(255))
    line_number = Column(Integer)
    extra_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Metric(Base):
    """Metrics model."""
    __tablename__ = "metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    value = Column(Float, nullable=False)
    tags = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    component = Column(String(255))
    instance = Column(String(255))


class APIKey(Base):
    """API Key model."""
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    user_id = Column(String, ForeignKey("users.id"))
    permissions = Column(JSON)
    rate_limit = Column(Integer, default=1000)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Production-grade database manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.async_session_factory = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize database connection and setup."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.config.connection_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            await self.create_tables()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created/verified")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper error handling."""
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            async with self.get_session() as session:
                result = await session.execute(select(1))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            async with self.get_session() as session:
                stats = {}
                
                # Count records in each table
                for table_name, table_class in [
                    ("users", User),
                    ("models", Model),
                    ("datasets", Dataset),
                    ("experiments", Experiment),
                    ("training_jobs", TrainingJob),
                    ("inference_jobs", InferenceJob),
                    ("system_logs", SystemLog),
                    ("metrics", Metric),
                    ("api_keys", APIKey)
                ]:
                    result = await session.execute(
                        select(table_class).count()
                    )
                    stats[f"{table_name}_count"] = result.scalar()
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def cleanup_old_records(self, days: int = 30):
        """Clean up old records."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.get_session() as session:
                # Clean old logs
                await session.execute(
                    delete(SystemLog).where(SystemLog.timestamp < cutoff_date)
                )
                
                # Clean old metrics
                await session.execute(
                    delete(Metric).where(Metric.timestamp < cutoff_date)
                )
                
                # Clean completed inference jobs
                await session.execute(
                    delete(InferenceJob).where(
                        InferenceJob.completed_at < cutoff_date
                    )
                )
                
                await session.commit()
                self.logger.info(f"Cleaned up records older than {days} days")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
    
    async def backup_database(self, backup_path: str) -> bool:
        """Create database backup."""
        try:
            # This is a simplified backup - in production, use proper backup tools
            if self.config.database_type == DatabaseType.POSTGRESQL:
                # Use pg_dump
                cmd = [
                    "pg_dump",
                    "-h", self.config.host,
                    "-p", str(self.config.port),
                    "-U", self.config.username,
                    "-d", self.config.database,
                    "-f", backup_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.logger.info(f"Database backup created: {backup_path}")
                    return True
                else:
                    self.logger.error(f"Backup failed: {stderr.decode()}")
                    return False
            
            elif self.config.database_type == DatabaseType.SQLITE:
                # Copy SQLite file
                import shutil
                shutil.copy2(f"{self.config.database}.db", backup_path)
                self.logger.info(f"SQLite database copied to: {backup_path}")
                return True
            
            else:
                self.logger.warning(f"Backup not implemented for {self.config.database_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    async def run_migrations(self):
        """Run database migrations using Alembic."""
        try:
            # Create Alembic configuration
            alembic_cfg = Config(f"{self.config.migration_path}/alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.connection_url)
            
            # Run migrations
            command.upgrade(alembic_cfg, "head")
            self.logger.info("Database migrations completed")
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connections closed")


class UserRepository:
    """User data access layer."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[User]:
        """Create a new user."""
        try:
            async with self.db_manager.get_session() as session:
                user = User(**user_data)
                session.add(user)
                await session.flush()  # Get the ID
                await session.refresh(user)
                return user
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Failed to get user by ID: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(User).where(User.username == username)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Failed to get user by username: {e}")
            return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user."""
        try:
            async with self.db_manager.get_session() as session:
                await session.execute(
                    update(User).where(User.id == user_id).values(**updates)
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to update user: {e}")
            return False
    
    async def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List users with pagination."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(User).limit(limit).offset(offset)
                )
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to list users: {e}")
            return []


class ModelRepository:
    """Model data access layer."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_model(self, model_data: Dict[str, Any]) -> Optional[Model]:
        """Create a new model."""
        try:
            async with self.db_manager.get_session() as session:
                model = Model(**model_data)
                session.add(model)
                await session.flush()
                await session.refresh(model)
                return model
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            return None
    
    async def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Get model by ID."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(Model).where(Model.id == model_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Failed to get model by ID: {e}")
            return None
    
    async def list_models_by_status(self, status: ModelStatus) -> List[Model]:
        """List models by status."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(Model).where(Model.status == status.value)
                )
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to list models by status: {e}")
            return []
    
    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status."""
        try:
            async with self.db_manager.get_session() as session:
                await session.execute(
                    update(Model)
                    .where(Model.id == model_id)
                    .values(status=status.value, updated_at=datetime.utcnow())
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False


class TrainingJobRepository:
    """Training job data access layer."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_training_job(self, job_data: Dict[str, Any]) -> Optional[TrainingJob]:
        """Create a new training job."""
        try:
            async with self.db_manager.get_session() as session:
                job = TrainingJob(**job_data)
                session.add(job)
                await session.flush()
                await session.refresh(job)
                return job
        except Exception as e:
            self.logger.error(f"Failed to create training job: {e}")
            return None
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: TrainingStatus, 
        progress: float = None,
        error_message: str = None
    ) -> bool:
        """Update training job status."""
        try:
            updates = {
                "status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if progress is not None:
                updates["progress"] = progress
            
            if error_message:
                updates["error_message"] = error_message
            
            if status == TrainingStatus.RUNNING and "start_time" not in updates:
                updates["start_time"] = datetime.utcnow()
            elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                updates["end_time"] = datetime.utcnow()
            
            async with self.db_manager.get_session() as session:
                await session.execute(
                    update(TrainingJob).where(TrainingJob.id == job_id).values(**updates)
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to update training job status: {e}")
            return False
    
    async def get_active_jobs(self) -> List[TrainingJob]:
        """Get active training jobs."""
        try:
            async with self.db_manager.get_session() as session:
                result = await session.execute(
                    select(TrainingJob).where(
                        TrainingJob.status.in_([
                            TrainingStatus.PENDING.value,
                            TrainingStatus.RUNNING.value
                        ])
                    )
                )
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to get active jobs: {e}")
            return []


class MetricsRepository:
    """Metrics data access layer."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def record_metric(
        self, 
        name: str, 
        value: float, 
        tags: Dict[str, str] = None,
        component: str = None,
        instance: str = None
    ) -> bool:
        """Record a metric."""
        try:
            async with self.db_manager.get_session() as session:
                metric = Metric(
                    name=name,
                    value=value,
                    tags=tags or {},
                    component=component,
                    instance=instance
                )
                session.add(metric)
                return True
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
            return False
    
    async def get_metrics(
        self, 
        name: str = None,
        component: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[Metric]:
        """Get metrics with filters."""
        try:
            async with self.db_manager.get_session() as session:
                query = select(Metric)
                
                if name:
                    query = query.where(Metric.name == name)
                if component:
                    query = query.where(Metric.component == component)
                if start_time:
                    query = query.where(Metric.timestamp >= start_time)
                if end_time:
                    query = query.where(Metric.timestamp <= end_time)
                
                query = query.order_by(Metric.timestamp.desc()).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return []


class DataAccessLayer:
    """
    Production-grade data access layer for Symbio AI.
    
    Provides unified interface to database operations with connection pooling,
    transaction management, and repository pattern implementation.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.user_repo = UserRepository(self.db_manager)
        self.model_repo = ModelRepository(self.db_manager)
        self.training_repo = TrainingJobRepository(self.db_manager)
        self.metrics_repo = MetricsRepository(self.db_manager)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize data access layer."""
        await self.db_manager.initialize()
        self.logger.info("Data access layer initialized")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        db_healthy = await self.db_manager.health_check()
        stats = await self.db_manager.get_stats()
        
        return {
            "database_healthy": db_healthy,
            "connection_url": self.config.connection_url.split('@')[0] + "@***",  # Hide credentials
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.db_manager.close()
        self.logger.info("Data access layer cleaned up")
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get repository configuration info."""
        return {
            "database_type": self.config.database_type.value,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "repositories": [
                "user_repository",
                "model_repository", 
                "training_job_repository",
                "metrics_repository"
            ],
            "features": [
                "connection_pooling",
                "async_operations",
                "transaction_management",
                "automatic_migrations",
                "health_monitoring",
                "backup_support"
            ]
        }