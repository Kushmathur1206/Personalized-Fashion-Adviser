"""
Advanced Recommendation Engine for Personalized Fashion Adviser.

Modules:
- engine: Multi-modal recommendation with weighted similarity and LSTM sequence model
- diversity: Diversity-aware re-ranking for varied recommendations
- filters: Category and attribute-based filtering
"""

from recommendation.engine import RecommendationEngine
from recommendation.diversity import DiversityReranker
from recommendation.filters import AttributeFilter
