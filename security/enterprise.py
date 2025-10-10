"""
Advanced security and encryption module for Symbio AI.

Provides enterprise-grade security features including authentication,
authorization, encryption, and compliance frameworks.
"""

import asyncio
import hashlib
import hmac
import secrets
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
from pathlib import Path
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import uuid
import time
import re


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    MODEL_TRAIN = "model_train"
    MODEL_DEPLOY = "model_deploy"
    DATA_ACCESS = "data_access"
    SYSTEM_MONITOR = "system_monitor"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: List[Permission]
    security_level: SecurityLevel
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_claims: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessToken:
    """JWT access token."""
    token: str
    expires_at: datetime
    token_type: str = "Bearer"
    scopes: List[str] = field(default_factory=list)


@dataclass
class ApiKey:
    """API key for service authentication."""
    key_id: str
    key_hash: str
    name: str
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None
    enabled: bool = True


class EncryptionProvider(ABC):
    """Abstract encryption provider."""
    
    @abstractmethod
    async def encrypt(self, data: bytes, key_id: str = None) -> Tuple[bytes, str]:
        """Encrypt data and return encrypted bytes with key ID."""
        pass
    
    @abstractmethod
    async def decrypt(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key."""
        pass
    
    @abstractmethod
    async def generate_key(self, key_id: str = None) -> str:
        """Generate a new encryption key."""
        pass


class AESEncryptionProvider(EncryptionProvider):
    """AES-256-GCM encryption provider."""
    
    def __init__(self):
        self.keys: Dict[str, bytes] = {}
        self.default_key_id = "default"
        self.logger = logging.getLogger(__name__)
        
        # Generate default key
        asyncio.create_task(self.generate_key(self.default_key_id))
    
    async def encrypt(self, data: bytes, key_id: str = None) -> Tuple[bytes, str]:
        """Encrypt data using AES-256-GCM."""
        if key_id is None:
            key_id = self.default_key_id
        
        if key_id not in self.keys:
            raise ValueError(f"Encryption key {key_id} not found")
        
        key = self.keys[key_id]
        
        # Generate random IV
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine IV + ciphertext + auth_tag
        encrypted_data = iv + ciphertext + encryptor.tag
        
        return encrypted_data, key_id
    
    async def decrypt(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        if key_id not in self.keys:
            raise ValueError(f"Decryption key {key_id} not found")
        
        key = self.keys[key_id]
        
        # Extract components
        iv = encrypted_data[:12]
        ciphertext = encrypted_data[12:-16]
        tag = encrypted_data[-16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    async def generate_key(self, key_id: str = None) -> str:
        """Generate new AES-256 key."""
        if key_id is None:
            key_id = str(uuid.uuid4())
        
        key = secrets.token_bytes(32)  # 256-bit key
        self.keys[key_id] = key
        
        self.logger.info(f"Generated encryption key: {key_id}")
        return key_id


class RSAEncryptionProvider(EncryptionProvider):
    """RSA encryption provider for asymmetric encryption."""
    
    def __init__(self):
        self.key_pairs: Dict[str, Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def encrypt(self, data: bytes, key_id: str = None) -> Tuple[bytes, str]:
        """Encrypt data using RSA public key."""
        if key_id not in self.key_pairs:
            raise ValueError(f"RSA key pair {key_id} not found")
        
        _, public_key = self.key_pairs[key_id]
        
        # RSA can only encrypt small amounts of data
        # For larger data, use hybrid encryption (RSA + AES)
        if len(data) > 190:  # RSA-2048 can encrypt max ~245 bytes with OAEP
            # Generate AES key
            aes_key = secrets.token_bytes(32)
            
            # Encrypt data with AES
            aes_provider = AESEncryptionProvider()
            aes_provider.keys["temp"] = aes_key
            encrypted_data, _ = await aes_provider.encrypt(data, "temp")
            
            # Encrypt AES key with RSA
            encrypted_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key + encrypted data
            result = len(encrypted_key).to_bytes(4, 'big') + encrypted_key + encrypted_data
            return result, key_id
        else:
            # Direct RSA encryption for small data
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted_data, key_id
    
    async def decrypt(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt RSA encrypted data."""
        if key_id not in self.key_pairs:
            raise ValueError(f"RSA key pair {key_id} not found")
        
        private_key, _ = self.key_pairs[key_id]
        
        # Check if this is hybrid encryption
        if len(encrypted_data) > 256:  # Likely hybrid encryption
            # Extract encrypted AES key length
            key_length = int.from_bytes(encrypted_data[:4], 'big')
            encrypted_aes_key = encrypted_data[4:4+key_length]
            aes_encrypted_data = encrypted_data[4+key_length:]
            
            # Decrypt AES key
            aes_key = private_key.decrypt(
                encrypted_aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            aes_provider = AESEncryptionProvider()
            aes_provider.keys["temp"] = aes_key
            plaintext = await aes_provider.decrypt(aes_encrypted_data, "temp")
            return plaintext
        else:
            # Direct RSA decryption
            plaintext = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return plaintext
    
    async def generate_key(self, key_id: str = None) -> str:
        """Generate new RSA key pair."""
        if key_id is None:
            key_id = str(uuid.uuid4())
        
        # Generate 2048-bit RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        self.key_pairs[key_id] = (private_key, public_key)
        
        self.logger.info(f"Generated RSA key pair: {key_id}")
        return key_id


class AuthenticationManager:
    """Manages user authentication and session management."""
    
    def __init__(self, secret_key: str, token_expiry: int = 3600):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.api_keys: Dict[str, ApiKey] = {}
        self.blacklisted_tokens: set = set()
        self.logger = logging.getLogger(__name__)
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str = None,
        user_agent: str = None
    ) -> Optional[AccessToken]:
        """Authenticate user with username/password."""
        # In production, this would verify against a user database
        # For demo purposes, using hardcoded admin user
        if username == "admin" and password == "symbio_ai_admin_2025":
            return await self._create_access_token(
                user_id=username,
                permissions=[Permission.ADMIN],
                security_level=SecurityLevel.SECRET,
                ip_address=ip_address,
                user_agent=user_agent
            )
        
        self.logger.warning(f"Failed authentication attempt for user: {username}")
        return None
    
    async def authenticate_api_key(self, api_key: str) -> Optional[SecurityContext]:
        """Authenticate using API key."""
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Find matching API key
        for key_id, stored_key in self.api_keys.items():
            if stored_key.key_hash == key_hash and stored_key.enabled:
                # Check expiration
                if stored_key.expires_at and datetime.now() > stored_key.expires_at:
                    continue
                
                # Check rate limit
                if stored_key.rate_limit and stored_key.usage_count >= stored_key.rate_limit:
                    continue
                
                # Update usage
                stored_key.last_used = datetime.now()
                stored_key.usage_count += 1
                
                # Create security context
                context = SecurityContext(
                    user_id=f"api_key_{key_id}",
                    session_id=str(uuid.uuid4()),
                    permissions=stored_key.permissions,
                    security_level=SecurityLevel.INTERNAL,
                    expires_at=datetime.now() + timedelta(seconds=3600)
                )
                
                self.active_sessions[context.session_id] = context
                return context
        
        self.logger.warning("Invalid API key authentication attempt")
        return None
    
    async def _create_access_token(
        self,
        user_id: str,
        permissions: List[Permission],
        security_level: SecurityLevel,
        ip_address: str = None,
        user_agent: str = None
    ) -> AccessToken:
        """Create JWT access token."""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=self.token_expiry)
        
        # Create security context
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            permissions=permissions,
            security_level=security_level,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store session
        self.active_sessions[session_id] = context
        
        # Create JWT payload
        payload = {
            'user_id': user_id,
            'session_id': session_id,
            'permissions': [p.value for p in permissions],
            'security_level': security_level.value,
            'exp': expires_at.timestamp(),
            'iat': datetime.now().timestamp(),
            'jti': str(uuid.uuid4())
        }
        
        # Sign JWT
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        return AccessToken(
            token=token,
            expires_at=expires_at,
            scopes=[p.value for p in permissions]
        )
    
    async def verify_token(self, token: str) -> Optional[SecurityContext]:
        """Verify JWT token and return security context."""
        if token in self.blacklisted_tokens:
            return None
        
        try:
            # Decode JWT
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            # Check if session exists and is valid
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                if datetime.now() < context.expires_at:
                    return context
                else:
                    # Session expired
                    del self.active_sessions[session_id]
            
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
        
        return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke (blacklist) a token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            # Add to blacklist
            self.blacklisted_tokens.add(token)
            
            # Remove session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            return True
        except jwt.InvalidTokenError:
            return False
    
    async def create_api_key(
        self,
        name: str,
        permissions: List[Permission],
        expires_days: int = None,
        rate_limit: int = None
    ) -> Tuple[str, str]:
        """Create new API key."""
        key_id = str(uuid.uuid4())
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        api_key_obj = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at,
            rate_limit=rate_limit
        )
        
        self.api_keys[key_id] = api_key_obj
        
        self.logger.info(f"Created API key: {name} (ID: {key_id})")
        return api_key, key_id
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self.api_keys:
            self.api_keys[key_id].enabled = False
            self.logger.info(f"Revoked API key: {key_id}")
            return True
        return False


class AuthorizationManager:
    """Manages access control and authorization policies."""
    
    def __init__(self):
        self.resource_policies: Dict[str, Dict[str, Any]] = {}
        self.role_permissions: Dict[str, List[Permission]] = {
            "admin": list(Permission),
            "operator": [
                Permission.READ, Permission.WRITE, Permission.EXECUTE,
                Permission.MODEL_DEPLOY, Permission.SYSTEM_MONITOR
            ],
            "scientist": [
                Permission.READ, Permission.WRITE, Permission.MODEL_TRAIN,
                Permission.DATA_ACCESS
            ],
            "viewer": [Permission.READ, Permission.SYSTEM_MONITOR]
        }
        self.logger = logging.getLogger(__name__)
    
    async def check_permission(
        self,
        context: SecurityContext,
        resource: str,
        action: Permission,
        resource_data: Dict[str, Any] = None
    ) -> bool:
        """Check if user has permission to perform action on resource."""
        # Check basic permission
        if action not in context.permissions:
            return False
        
        # Check resource-specific policies
        if resource in self.resource_policies:
            policy = self.resource_policies[resource]
            
            # Check security level requirement
            required_level = policy.get('min_security_level')
            if required_level:
                required_enum = SecurityLevel(required_level)
                if context.security_level.value < required_enum.value:
                    return False
            
            # Check custom policy function
            policy_func = policy.get('policy_function')
            if policy_func:
                try:
                    result = await policy_func(context, action, resource_data or {})
                    if not result:
                        return False
                except Exception as e:
                    self.logger.error(f"Policy function error: {e}")
                    return False
        
        return True
    
    def add_resource_policy(
        self,
        resource: str,
        min_security_level: SecurityLevel = None,
        policy_function: Callable = None,
        additional_rules: Dict[str, Any] = None
    ) -> None:
        """Add access policy for a resource."""
        policy = {}
        
        if min_security_level:
            policy['min_security_level'] = min_security_level.value
        
        if policy_function:
            policy['policy_function'] = policy_function
        
        if additional_rules:
            policy.update(additional_rules)
        
        self.resource_policies[resource] = policy
        self.logger.info(f"Added resource policy for: {resource}")
    
    async def get_user_resources(self, context: SecurityContext) -> List[str]:
        """Get list of resources user has access to."""
        accessible_resources = []
        
        for resource, policy in self.resource_policies.items():
            # Check basic access
            if await self.check_permission(context, resource, Permission.READ):
                accessible_resources.append(resource)
        
        return accessible_resources


class DataClassificationManager:
    """Manages data classification and protection policies."""
    
    def __init__(self, encryption_provider: EncryptionProvider):
        self.encryption_provider = encryption_provider
        self.classification_policies: Dict[SecurityLevel, Dict[str, Any]] = {
            SecurityLevel.PUBLIC: {
                'encryption_required': False,
                'access_logging': False,
                'retention_days': 365
            },
            SecurityLevel.INTERNAL: {
                'encryption_required': True,
                'access_logging': True,
                'retention_days': 180
            },
            SecurityLevel.CONFIDENTIAL: {
                'encryption_required': True,
                'access_logging': True,
                'retention_days': 90,
                'audit_trail': True
            },
            SecurityLevel.SECRET: {
                'encryption_required': True,
                'access_logging': True,
                'retention_days': 30,
                'audit_trail': True,
                'multi_factor_auth': True
            },
            SecurityLevel.TOP_SECRET: {
                'encryption_required': True,
                'access_logging': True,
                'retention_days': 7,
                'audit_trail': True,
                'multi_factor_auth': True,
                'approval_required': True
            }
        }
        self.logger = logging.getLogger(__name__)
    
    async def classify_data(self, data: bytes, patterns: List[str] = None) -> SecurityLevel:
        """Automatically classify data based on content patterns."""
        if patterns is None:
            patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{16}\b',  # Credit card pattern
                r'\bpassword\b',  # Password fields
                r'\bapi[_-]?key\b',  # API keys
                r'\bsecret\b'  # Secret data
            ]
        
        data_str = data.decode('utf-8', errors='ignore').lower()
        
        # Check for sensitive patterns
        for pattern in patterns:
            if re.search(pattern, data_str):
                return SecurityLevel.CONFIDENTIAL
        
        # Default classification
        return SecurityLevel.INTERNAL
    
    async def protect_data(
        self,
        data: bytes,
        classification: SecurityLevel,
        context: SecurityContext
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Apply protection policies to data based on classification."""
        policy = self.classification_policies[classification]
        metadata = {
            'classification': classification.value,
            'protected_at': datetime.now().isoformat(),
            'protected_by': context.user_id
        }
        
        # Apply encryption if required
        if policy.get('encryption_required', False):
            encrypted_data, key_id = await self.encryption_provider.encrypt(data)
            metadata['encrypted'] = True
            metadata['key_id'] = key_id
            data = encrypted_data
        
        # Log access if required
        if policy.get('access_logging', False):
            self.logger.info(
                f"Data access: {context.user_id} accessed {classification.value} data",
                extra={
                    'user_id': context.user_id,
                    'classification': classification.value,
                    'data_size': len(data),
                    'session_id': context.session_id
                }
            )
        
        return data, metadata


class ComplianceManager:
    """Manages regulatory compliance (GDPR, CCPA, SOX, etc.)."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
        self.data_retention_policies: Dict[str, int] = {}  # resource -> days
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def log_audit_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Log audit event for compliance."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'result': result,
            'ip_address': details.get('ip_address') if details else None,
            'user_agent': details.get('user_agent') if details else None,
            'details': details or {}
        }
        
        self.audit_log.append(audit_entry)
        
        # In production, this would be sent to a secure audit system
        self.logger.info(f"AUDIT: {event_type} - {user_id} {action} {resource}: {result}")
    
    async def check_data_retention(self, resource: str, created_at: datetime) -> bool:
        """Check if data meets retention policy requirements."""
        if resource not in self.data_retention_policies:
            return True  # No policy means keep indefinitely
        
        retention_days = self.data_retention_policies[resource]
        expiry_date = created_at + timedelta(days=retention_days)
        
        return datetime.now() < expiry_date
    
    async def record_consent(
        self,
        user_id: str,
        purpose: str,
        consent_given: bool,
        legal_basis: str = None
    ) -> str:
        """Record user consent for GDPR compliance."""
        consent_id = str(uuid.uuid4())
        consent_record = {
            'consent_id': consent_id,
            'user_id': user_id,
            'purpose': purpose,
            'consent_given': consent_given,
            'legal_basis': legal_basis,
            'recorded_at': datetime.now().isoformat(),
            'ip_address': None,  # Would be captured from request context
            'method': 'api'
        }
        
        self.consent_records[consent_id] = consent_record
        
        await self.log_audit_event(
            'consent_recorded',
            user_id,
            f'consent_{consent_id}',
            'record_consent',
            'success',
            consent_record
        )
        
        return consent_id
    
    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get all data associated with a user (GDPR data portability)."""
        user_data = {
            'user_id': user_id,
            'exported_at': datetime.now().isoformat(),
            'consent_records': [
                record for record in self.consent_records.values()
                if record['user_id'] == user_id
            ],
            'audit_events': [
                event for event in self.audit_log
                if event['user_id'] == user_id
            ]
        }
        
        await self.log_audit_event(
            'data_export',
            user_id,
            f'user_data_{user_id}',
            'export_data',
            'success'
        )
        
        return user_data
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data (GDPR right to erasure)."""
        try:
            # Remove consent records
            consent_ids_to_remove = [
                consent_id for consent_id, record in self.consent_records.items()
                if record['user_id'] == user_id
            ]
            
            for consent_id in consent_ids_to_remove:
                del self.consent_records[consent_id]
            
            # Anonymize audit logs (can't delete for compliance)
            for event in self.audit_log:
                if event['user_id'] == user_id:
                    event['user_id'] = 'anonymized'
            
            await self.log_audit_event(
                'data_deletion',
                user_id,
                f'user_data_{user_id}',
                'delete_data',
                'success'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete user data for {user_id}: {e}")
            return False


class SecurityManager:
    """
    Production-grade security manager for Symbio AI.
    
    Integrates authentication, authorization, encryption, and compliance
    into a unified security framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize security components
        secret_key = config.get('secret_key', secrets.token_urlsafe(32))
        self.auth_manager = AuthenticationManager(secret_key)
        self.authz_manager = AuthorizationManager()
        
        # Initialize encryption providers
        self.aes_provider = AESEncryptionProvider()
        self.rsa_provider = RSAEncryptionProvider()
        
        # Choose default encryption provider
        encryption_type = config.get('encryption_type', 'aes')
        if encryption_type == 'rsa':
            self.encryption_provider = self.rsa_provider
        else:
            self.encryption_provider = self.aes_provider
        
        self.classification_manager = DataClassificationManager(self.encryption_provider)
        self.compliance_manager = ComplianceManager()
        
        self.logger = logging.getLogger(__name__)
        
        # Setup default security policies
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default security policies."""
        # Model training requires CONFIDENTIAL level
        async def model_training_policy(context, action, resource_data):
            return context.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]
        
        self.authz_manager.add_resource_policy(
            'model_training',
            SecurityLevel.CONFIDENTIAL,
            model_training_policy
        )
        
        # System administration requires SECRET level
        self.authz_manager.add_resource_policy(
            'system_admin',
            SecurityLevel.SECRET
        )
        
        # Data access policies
        async def sensitive_data_policy(context, action, resource_data):
            classification = resource_data.get('classification', 'internal')
            if classification in ['secret', 'top_secret']:
                return context.security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]
            return True
        
        self.authz_manager.add_resource_policy(
            'sensitive_data',
            SecurityLevel.INTERNAL,
            sensitive_data_policy
        )
    
    async def authenticate(
        self,
        credentials: Dict[str, str],
        auth_type: str = "password",
        request_info: Dict[str, str] = None
    ) -> Optional[AccessToken]:
        """Authenticate user with various methods."""
        request_info = request_info or {}
        
        if auth_type == "password":
            return await self.auth_manager.authenticate_user(
                credentials.get('username'),
                credentials.get('password'),
                request_info.get('ip_address'),
                request_info.get('user_agent')
            )
        elif auth_type == "api_key":
            context = await self.auth_manager.authenticate_api_key(
                credentials.get('api_key')
            )
            if context:
                # Convert context to token format
                return AccessToken(
                    token=f"api_key_{context.session_id}",
                    expires_at=context.expires_at
                )
        
        return None
    
    async def authorize(
        self,
        token: str,
        resource: str,
        action: Permission,
        resource_data: Dict[str, Any] = None
    ) -> bool:
        """Check if token holder is authorized for action."""
        # Verify token and get context
        context = await self.auth_manager.verify_token(token)
        if not context:
            return False
        
        # Check authorization
        authorized = await self.authz_manager.check_permission(
            context, resource, action, resource_data
        )
        
        # Log authorization attempt
        await self.compliance_manager.log_audit_event(
            'authorization_check',
            context.user_id,
            resource,
            action.value,
            'success' if authorized else 'denied',
            {
                'session_id': context.session_id,
                'security_level': context.security_level.value
            }
        )
        
        return authorized
    
    async def protect_data(
        self,
        data: bytes,
        context: SecurityContext,
        classification: SecurityLevel = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Protect data with appropriate security measures."""
        # Auto-classify if not provided
        if classification is None:
            classification = await self.classification_manager.classify_data(data)
        
        # Apply protection
        protected_data, metadata = await self.classification_manager.protect_data(
            data, classification, context
        )
        
        return protected_data, metadata
    
    async def create_api_key(
        self,
        context: SecurityContext,
        name: str,
        permissions: List[Permission],
        expires_days: int = None
    ) -> Optional[Tuple[str, str]]:
        """Create API key (requires admin permission)."""
        if Permission.ADMIN not in context.permissions:
            return None
        
        api_key, key_id = await self.auth_manager.create_api_key(
            name, permissions, expires_days
        )
        
        await self.compliance_manager.log_audit_event(
            'api_key_created',
            context.user_id,
            f'api_key_{key_id}',
            'create',
            'success',
            {
                'key_name': name,
                'permissions': [p.value for p in permissions],
                'expires_days': expires_days
            }
        )
        
        return api_key, key_id
    
    async def encrypt_model_data(self, model_data: bytes) -> Tuple[bytes, str]:
        """Encrypt model data for secure storage."""
        return await self.encryption_provider.encrypt(model_data)
    
    async def decrypt_model_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt model data."""
        return await self.encryption_provider.decrypt(encrypted_data, key_id)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary."""
        return {
            'active_sessions': len(self.auth_manager.active_sessions),
            'api_keys': len(self.auth_manager.api_keys),
            'blacklisted_tokens': len(self.auth_manager.blacklisted_tokens),
            'resource_policies': len(self.authz_manager.resource_policies),
            'audit_events': len(self.compliance_manager.audit_log),
            'consent_records': len(self.compliance_manager.consent_records),
            'encryption_keys': len(self.aes_provider.keys) + len(self.rsa_provider.key_pairs)
        }