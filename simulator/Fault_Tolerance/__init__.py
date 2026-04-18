#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FT-Oriented 容错机制模块
基于显式冗余组的PIM硬件容错框架
"""

__version__ = "1.0.0"
__author__ = "FT-Oriented Grouping Team"

# 导出主要接口
from .fault_tolerance_simulation import FaultToleranceSimulator
from .pattern_data_loader import PatternDataLoader
from .redundancy_group_parser import RedundancyGroupParser
from .fault_injector import FaultInjector
from .majority_voter import MajorityVoter
from .metrics_collector import MetricsCollector

__all__ = [
    'FaultToleranceSimulator',
    'PatternDataLoader',
    'RedundancyGroupParser',
    'FaultInjector',
    'MajorityVoter',
    'MetricsCollector',
]

print("🛡️ FT-Oriented 容错机制模块已加载")
