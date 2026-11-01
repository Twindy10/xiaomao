"""语义缓存。"""

from xiaomao.cache.policy import should_cache
from xiaomao.cache.semantic import REUSE_QUESTION, CacheHit, SemanticCache

__all__ = ["CacheHit", "SemanticCache", "REUSE_QUESTION", "should_cache"]
