import os
import json
import time
import logging
from typing import Any, Dict, Optional
import threading

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages in-memory cache with disk persistence."""
    
    def __init__(self, cache_file: str, max_size: int = 500, default_expiry: int = 3600):
        self.cache_file = cache_file
        self.max_size = max_size
        self.default_expiry = default_expiry
        self._cache = {}  # {key: value}
        self._timestamps = {}  # {key: timestamp}
        self._expiry = {}  # {key: expiry_seconds}
        self._lock = threading.RLock()
        self._save_interval = 300  # Save every 5 minutes
        self._last_save = 0
        self._load_from_disk()
        self._start_save_thread()
    
    def _load_from_disk(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    with self._lock:
                        self._cache = data.get('cache', {})
                        self._timestamps = {k: float(v) for k, v in data.get('timestamps', {}).items()}
                        self._expiry = {k: int(v) for k, v in data.get('expiry', {}).items()}
                logger.info(f"Loaded {len(self._cache)} cached items from disk")
        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")
            # Start with empty cache
            with self._lock:
                self._cache = {}
                self._timestamps = {}
                self._expiry = {}
    
    def _save_to_disk(self):
        """Save cache to disk."""
        try:
            # Skip if no changes since last save
            now = time.time()
            if now - self._last_save < self._save_interval and self._last_save > 0:
                return
            
            # Prepare cache data
            with self._lock:
                # Perform cleanup first
                self._cleanup()
                
                data = {
                    'cache': self._cache,
                    'timestamps': self._timestamps,
                    'expiry': self._expiry
                }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Save to file
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
            
            self._last_save = now
            logger.info(f"Saved {len(self._cache)} cached items to disk")
        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")
    
    def _start_save_thread(self):
        """Start a thread to periodically save cache to disk."""
        def save_loop():
            while True:
                time.sleep(self._save_interval)
                self._save_to_disk()
        
        thread = threading.Thread(target=save_loop, daemon=True)
        thread.start()
    
    def _cleanup(self):
        """Remove expired items and enforce max size."""
        now = time.time()
        
        # Remove expired items
        expired_keys = [k for k, timestamp in self._timestamps.items() 
                       if now - timestamp > self._expiry.get(k, self.default_expiry)]
        
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]
            if key in self._expiry:
                del self._expiry[key]
        
        # Enforce max size
        if len(self._cache) > self.max_size:
            # Sort keys by timestamp (oldest first)
            sorted_keys = sorted(self._timestamps.items(), key=lambda x: x[1])
            # Remove oldest items until we're under max_size
            keys_to_remove = [k for k, _ in sorted_keys[:len(sorted_keys) - self.max_size]]
            
            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                if key in self._expiry:
                    del self._expiry[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if it exists and is not expired."""
        with self._lock:
            if key not in self._cache or key not in self._timestamps:
                return None
            
            now = time.time()
            timestamp = self._timestamps[key]
            expiry = self._expiry.get(key, self.default_expiry)
            
            if now - timestamp > expiry:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
                if key in self._expiry:
                    del self._expiry[key]
                return None
            
            return self._cache[key]
    
    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> None:
        """Store a value in cache with optional expiry time."""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
            if expiry is not None:
                self._expiry[key] = expiry
            
            # Trigger cleanup if we've exceeded max_size
            if len(self._cache) > self.max_size:
                self._cleanup()
                
            # Save if it's been a while since last save
            now = time.time()
            if now - self._last_save > self._save_interval:
                # Use a separate thread to avoid blocking
                threading.Thread(target=self._save_to_disk, daemon=True).start()
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                if key in self._expiry:
                    del self._expiry[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._expiry.clear()
        self._save_to_disk()
        
    def keys(self) -> list:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get number of items in cache."""
        with self._lock:
            return len(self._cache)