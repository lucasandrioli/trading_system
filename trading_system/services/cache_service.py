import logging
import os
import json
import time
import threading
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger("trading_system.services.cache_service")

class CacheService:
    """Service for handling in-memory and disk-based caching."""
    
    def __init__(self, config):
        """Initialize the cache service."""
        self.config = config
        self._cache = {}  # In-memory cache
        self._timestamps = {}  # Cache timestamps
        self._lock = threading.RLock()  # Thread safety lock
        self._save_thread = None
        self._running = True
        
        # Load cache from disk
        self._load_cache()
        
        # Start background thread for periodic cache saving
        self._start_save_thread()
        
        logger.debug("CacheService initialized")
    
    def get(self, key: str, expiry: Optional[int] = None) -> Any:
        """
        Get an item from cache.
        
        Args:
            key: Cache key
            expiry: Cache expiry in seconds (overrides default)
            
        Returns:
            Cached item or None if expired/not found
        """
        with self._lock:
            if expiry is None:
                expiry = self.config.get('CACHE_EXPIRY', 3600)  # Default 1 hour
            
            if key not in self._cache or key not in self._timestamps:
                return None
            
            timestamp = self._timestamps[key]
            now = time.time()
            
            if now - timestamp > expiry:
                # Cache expired
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
            # Limit cache size to prevent memory issues
            self._check_cache_size()
    
    def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            
            # Also clear cache file
            cache_file = os.path.join(self.config.get('DATA_FOLDER', 'data'), 'analysis_cache.json')
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass
            
            logger.info("Cache cleared")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get the current status of the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            try:
                # Calculate approximate memory usage
                sample_size = min(100, len(self._cache))
                if sample_size > 0:
                    sample_keys = list(self._cache.keys())[:sample_size]
                    sample_str = json.dumps({k: self._cache[k] for k in sample_keys})
                    avg_item_size = len(sample_str) / sample_size
                    total_items = len(self._cache)
                    estimated_size = (avg_item_size * total_items) / (1024 * 1024)  # MB
                else:
                    estimated_size = 0
                
                return {
                    "cache_size": len(self._cache),
                    "memory_usage_mb": estimated_size,
                    "items_count": len(self._cache),
                    "oldest_item_age": time.time() - min(self._timestamps.values()) if self._timestamps else None,
                    "newest_item_age": time.time() - max(self._timestamps.values()) if self._timestamps else None,
                    "categories": self._get_category_stats()
                }
            except Exception as e:
                logger.error(f"Error getting cache status: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "cache_size": len(self._cache)
                }
    
    def _get_category_stats(self) -> Dict[str, int]:
        """Get statistics on cache categories."""
        categories = {}
        
        for key in self._cache.keys():
            category = key.split(':')[0] if ':' in key else 'other'
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def _check_cache_size(self) -> None:
        """Check and limit cache size if necessary."""
        max_items = 1000  # Maximum items in cache
        
        if len(self._cache) > max_items:
            # Keep only the most recent items
            items_to_keep = 800  # Keep 80% of max
            
            # Sort by timestamp (newest first)
            sorted_items = sorted(self._timestamps.items(), key=lambda x: x[1], reverse=True)
            keep_keys = [k for k, _ in sorted_items[:items_to_keep]]
            
            # Create new dictionaries with only the keys to keep
            new_cache = {k: self._cache[k] for k in keep_keys if k in self._cache}
            new_timestamps = {k: self._timestamps[k] for k in keep_keys if k in self._timestamps}
            
            # Replace the old dictionaries
            self._cache = new_cache
            self._timestamps = new_timestamps
            
            logger.info(f"Cache pruned to {len(self._cache)} items")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = os.path.join(self.config.get('DATA_FOLDER', 'data'), 'analysis_cache.json')
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self._cache = cache_data.get('cache', {})
                    self._timestamps = cache_data.get('timestamps', {})
                    
                    # Convert timestamp strings to floats if needed
                    for key, value in self._timestamps.items():
                        if isinstance(value, str):
                            try:
                                self._timestamps[key] = float(value)
                            except ValueError:
                                # If conversion fails, use current time
                                self._timestamps[key] = time.time()
                
                logger.info(f"Loaded {len(self._cache)} cached items")
        except Exception as e:
            logger.error(f"Error loading cache: {e}", exc_info=True)
            self._cache = {}
            self._timestamps = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        with self._lock:
            try:
                # Limit cache size before saving
                self._check_cache_size()
                
                cache_data = {
                    'cache': self._cache,
                    'timestamps': self._timestamps
                }
                
                # Ensure data directory exists
                data_folder = self.config.get('DATA_FOLDER', 'data')
                os.makedirs(data_folder, exist_ok=True)
                
                cache_file = os.path.join(data_folder, 'analysis_cache.json')
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                
                logger.info(f"Saved {len(self._cache)} items to cache")
            except Exception as e:
                logger.error(f"Error saving cache: {e}", exc_info=True)
    
    def _start_save_thread(self) -> None:
        """Start a background thread for periodic cache saving."""
        def _save_thread_func():
            while self._running:
                time.sleep(300)  # Save every 5 minutes
                try:
                    self._save_cache()
                except Exception as e:
                    logger.error(f"Error in cache save thread: {e}", exc_info=True)
        
        self._save_thread = threading.Thread(
            target=_save_thread_func,
            daemon=True,
            name="CacheSaveThread"
        )
        self._save_thread.start()
    
    def shutdown(self) -> None:
        """Shutdown the cache service gracefully."""
        self._running = False
        
        # Save cache one last time
        try:
            self._save_cache()
        except Exception as e:
            logger.error(f"Error saving cache during shutdown: {e}", exc_info=True)
        
        # Wait for save thread to terminate
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        logger.info("CacheService shutdown complete")