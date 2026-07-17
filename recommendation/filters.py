"""
Attribute-Based Filtering Module.

Allows users to filter recommendations by:
- Category (only show shoes, only show tops, etc.)
- Style (only formal, only casual, etc.)
- Season (only summer appropriate, etc.)
- Color preference
- Price range (if metadata available)

Filters are applied post-retrieval to narrow down candidates.
"""

import numpy as np


class AttributeFilter:
    """
    Filters recommendation candidates based on predicted attributes.

    Works with the FashionClassifier's predictions to enable
    fine-grained filtering of results.
    """

    def __init__(self):
        """Initialize with empty metadata store."""
        self.item_metadata = {}  # idx -> {category, styles, seasons, colors}

    def set_metadata(self, metadata_list):
        """
        Store metadata for all items in the catalog.

        Args:
            metadata_list: List of dicts with keys:
                           category, styles, seasons, colors
        """
        self.item_metadata = {i: m for i, m in enumerate(metadata_list)}

    def filter_by_category(self, candidate_indices, target_category):
        """
        Keep only items matching the target category.

        Args:
            candidate_indices: List of candidate item indices
            target_category: Category string to filter for

        Returns:
            Filtered list of indices
        """
        if not self.item_metadata:
            return candidate_indices

        filtered = []
        for idx in candidate_indices:
            if idx in self.item_metadata:
                if self.item_metadata[idx].get('category') == target_category:
                    filtered.append(idx)

        return filtered if filtered else candidate_indices  # Fallback to unfiltered

    def filter_by_style(self, candidate_indices, target_styles):
        """
        Keep items matching any of the target styles.

        Args:
            candidate_indices: List of candidate item indices
            target_styles: List of style strings to match

        Returns:
            Filtered list of indices
        """
        if not self.item_metadata:
            return candidate_indices

        target_set = set(target_styles)
        filtered = []

        for idx in candidate_indices:
            if idx in self.item_metadata:
                item_styles = set(self.item_metadata[idx].get('styles', []))
                if item_styles & target_set:  # Intersection
                    filtered.append(idx)

        return filtered if filtered else candidate_indices

    def filter_by_season(self, candidate_indices, target_seasons):
        """
        Keep items suitable for target seasons.

        Args:
            candidate_indices: List of candidate item indices
            target_seasons: List of season strings

        Returns:
            Filtered list of indices
        """
        if not self.item_metadata:
            return candidate_indices

        target_set = set(target_seasons)
        filtered = []

        for idx in candidate_indices:
            if idx in self.item_metadata:
                item_seasons = set(self.item_metadata[idx].get('seasons', []))
                if item_seasons & target_set or 'All-Season' in item_seasons:
                    filtered.append(idx)

        return filtered if filtered else candidate_indices

    def filter_by_color(self, candidate_indices, target_colors):
        """
        Keep items matching color preferences.

        Args:
            candidate_indices: List of candidate item indices
            target_colors: List of preferred color strings

        Returns:
            Filtered list of indices
        """
        if not self.item_metadata:
            return candidate_indices

        target_set = set(target_colors)
        filtered = []

        for idx in candidate_indices:
            if idx in self.item_metadata:
                item_colors = set(self.item_metadata[idx].get('colors', []))
                if item_colors & target_set:
                    filtered.append(idx)

        return filtered if filtered else candidate_indices

    def filter_exclude_category(self, candidate_indices, exclude_category):
        """
        Remove items of a specific category (for complementary recommendations).

        Args:
            candidate_indices: List of candidate item indices
            exclude_category: Category to exclude

        Returns:
            Filtered list without the excluded category
        """
        if not self.item_metadata:
            return candidate_indices

        filtered = []
        for idx in candidate_indices:
            if idx in self.item_metadata:
                if self.item_metadata[idx].get('category') != exclude_category:
                    filtered.append(idx)

        return filtered if filtered else candidate_indices

    def apply_filters(self, candidate_indices, filters_dict):
        """
        Apply multiple filters in sequence.

        Args:
            candidate_indices: Initial candidate list
            filters_dict: Dictionary with filter type as key and value as target
                         e.g., {'category': 'Topwear', 'styles': ['Casual']}

        Returns:
            Filtered list of indices
        """
        result = candidate_indices

        if 'category' in filters_dict and filters_dict['category']:
            result = self.filter_by_category(result, filters_dict['category'])

        if 'styles' in filters_dict and filters_dict['styles']:
            result = self.filter_by_style(result, filters_dict['styles'])

        if 'seasons' in filters_dict and filters_dict['seasons']:
            result = self.filter_by_season(result, filters_dict['seasons'])

        if 'colors' in filters_dict and filters_dict['colors']:
            result = self.filter_by_color(result, filters_dict['colors'])

        if 'exclude_category' in filters_dict and filters_dict['exclude_category']:
            result = self.filter_exclude_category(
                result, filters_dict['exclude_category']
            )

        return result
