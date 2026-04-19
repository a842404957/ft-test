#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压缩模式数据加载器
兼容现有接口，使用压缩的冗余映射表
"""

import torch
import numpy as np
import struct
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from simulator.Fault_Tolerance.pattern_data_loader import PatternDataLoader
from compressed_redundancy_mapper import CompressedRedundancyLoader


class CompressedPatternDataLoader(PatternDataLoader):
    """压缩模式数据加载器，继承自PatternDataLoader"""

    def __init__(self, model_name: str = 'Res50',
                 translate_name: str = 'ft_group_cluster_translate',
                 data_dir: str = './'):
        """
        初始化压缩模式数据加载器

        Args:
            model_name: 模型名称
            translate_name: 转换方法名称
            data_dir: 数据文件目录
        """
        # 调用父类初始化，但不加载大型数据文件
        super().__init__(model_name, translate_name, data_dir)

        # 压缩映射相关
        self.compressed_loader = None
        self.compressed_file = None
        self.use_compressed = False

        # 初始化轻量级数据
        self._init_lightweight_data()

    def _init_lightweight_data(self):
        """初始化轻量级数据（不加载大型张量）"""
        # 尝试加载压缩文件
        compressed_file = Path(self.data_dir) / f'{self.model_name}_compressed_redundancy_map.bin'

        if compressed_file.exists():
            self.compressed_file = str(compressed_file)
            self.compressed_loader = CompressedRedundancyLoader(self.compressed_file)

            if self.compressed_loader.load_header():
                self.use_compressed = True
                print(f"✅ 使用压缩映射表: {compressed_file}")
            else:
                self.compressed_loader = None
                self.use_compressed = False
        else:
            print(f"⚠️ 压缩文件不存在: {compressed_file}")
            self.use_compressed = False

        # 如果没有压缩文件，回退到原始加载方式（仅加载轻量级数据）
        if not self.use_compressed:
            print("📂 回退到原始数据加载模式（轻量级）")
            self._load_fallback_data()

    def _load_fallback_data(self):
        """加载回退数据（仅加载必要的小文件）"""
        try:
            # 优先加载覆盖率信息，旧重用率命名作为兼容层
            self.coverage_ratio_information = self._load_pkl_file(
                f'model_{self.model_name}_{self.translate_name}_coverage_ratio_information.pkl'
            )
            self.reuse_ratio_information = self.coverage_ratio_information
            if self.reuse_ratio_information is None:
                self.reuse_ratio_information = self._load_pkl_file(
                    f'model_{self.model_name}_{self.translate_name}_reuse_ratio_information.pkl',
                    alternative_name=f'model_{self.model_name}_shape_and_value_reuse_ratio_information.pkl'
                )

            # 构建层配置信息
            self._build_layer_config()

            print(f"✅ 回退模式加载完成")

        except Exception as e:
            print(f"❌ 回退模式加载失败: {e}")

    def load_all_data(self) -> bool:
        """
        重写加载方法，优先使用压缩数据

        Returns:
            bool: 是否成功加载
        """
        try:
            if self.use_compressed:
                print("📂 使用压缩冗余映射表模式")
                # 压缩模式下不需要加载大型张量
                return True
            else:
                print("📂 使用传统模式（仅轻量级数据）")
                # 传统模式下只加载必要的小文件
                return self.reuse_ratio_information is not None

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def get_layer_map(self, layer_name: str) -> Optional[np.ndarray]:
        """
        获取层映射信息（兼容接口）
        动态构建稀疏映射矩阵

        Args:
            layer_name: 层名称

        Returns:
            映射张量或None
        """
        if self.use_compressed and self.compressed_loader:
            return self._build_sparse_map_from_compressed(layer_name)
        else:
            # 回退模式：构建简单的单位映射
            return self._build_identity_map(layer_name)

    def get_layer_multiplier(self, layer_name: str) -> Optional[np.ndarray]:
        """
        获取层倍数信息（兼容接口）

        Args:
            layer_name: 层名称

        Returns:
            倍数张量或None
        """
        if self.use_compressed and self.compressed_loader:
            return self._build_multiplier_from_compressed(layer_name)
        else:
            # 回退模式：返回单位倍数
            return self._build_identity_multiplier(layer_name)

    def _build_sparse_map_from_compressed(self, layer_name: str) -> Optional[np.ndarray]:
        """
        从压缩数据构建稀疏映射矩阵

        Args:
            layer_name: 层名称

        Returns:
            稀疏映射矩阵
        """
        try:
            # 获取层的映射数据
            mappings = self.compressed_loader.get_layer_mappings(layer_name)
            if not mappings:
                return None

            # 获取层信息
            layer_info = self.compressed_loader.layer_index.get(layer_name)
            if not layer_info:
                return None

            in_channels = layer_info['in_channels']
            out_channels = layer_info['out_channels']

            # 构建稀疏映射矩阵 [in_ch, out_ch, 2]
            # 初始化为无效值 (-1)
            map_tensor = np.full((in_channels, out_channels, 2), -1, dtype=np.int32)

            # 填充有效映射
            for source_ou, target_ou, multiplier in mappings:
                # 将线性OU索引转换为2D坐标
                source_out = source_ou // in_channels
                source_in = source_ou % in_channels

                if (0 <= source_out < out_channels and
                    0 <= source_in < in_channels):

                    # 设置映射关系
                    map_tensor[source_in, source_out, 0] = source_ou      # 源索引
                    map_tensor[source_in, source_out, 1] = target_ou     # 目标索引

            return map_tensor

        except Exception as e:
            print(f"❌ 构建稀疏映射矩阵失败 {layer_name}: {e}")
            return None

    def _build_multiplier_from_compressed(self, layer_name: str) -> Optional[np.ndarray]:
        """
        从压缩数据构建倍数矩阵

        Args:
            layer_name: 层名称

        Returns:
            倍数矩阵
        """
        try:
            # 获取层的映射数据
            mappings = self.compressed_loader.get_layer_mappings(layer_name)
            if not mappings:
                return None

            # 获取层信息
            layer_info = self.compressed_loader.layer_index.get(layer_name)
            if not layer_info:
                return None

            in_channels = layer_info['in_channels']
            out_channels = layer_info['out_channels']

            # 构建倍数矩阵，默认为1.0（无倍数）
            mult_tensor = np.ones((out_channels, in_channels), dtype=np.float32)

            # 填充倍数信息
            for source_ou, target_ou, multiplier in mappings:
                # 将线性OU索引转换为2D坐标
                source_out = source_ou // in_channels
                source_in = source_ou % in_channels

                if (0 <= source_out < out_channels and
                    0 <= source_in < in_channels):

                    # 设置倍数
                    mult_tensor[source_out, source_in] = float(multiplier)

            return mult_tensor

        except Exception as e:
            print(f"❌ 构建倍数矩阵失败 {layer_name}: {e}")
            return None

    def _build_identity_map(self, layer_name: str) -> Optional[np.ndarray]:
        """
        构建单位映射矩阵（回退模式）

        Args:
            layer_name: 层名称

        Returns:
            单位映射矩阵
        """
        try:
            # 获取层的基本信息
            layer_idx = self._get_layer_index(layer_name)
            if layer_idx is None:
                return None

            # 使用配置中的层信息
            if hasattr(self, 'layer_config') and 'weight_names' in self.layer_config:
                if layer_name in self.layer_config['weight_names']:
                    idx = self.layer_config['weight_names'].index(layer_name)
                    layer_type = self.layer_config['layer_types'][idx]

                    if layer_type == 'conv':
                        # 卷积层，使用典型尺寸
                        in_ch, out_ch = 64, 64
                    elif layer_type == 'fc':
                        # 全连接层
                        in_ch, out_ch = 512, 10
                    else:
                        in_ch, out_ch = 64, 64
                else:
                    in_ch, out_ch = 64, 64
            else:
                in_ch, out_ch = 64, 64

            # 构建单位映射矩阵
            map_tensor = np.full((in_ch, out_ch, 2), -1, dtype=np.int32)

            for out_ch in range(out_ch):
                for in_ch in range(in_ch):
                    ou_idx = out_ch * in_ch + in_ch
                    map_tensor[in_ch, out_ch, 0] = ou_idx  # 源索引
                    map_tensor[in_ch, out_ch, 1] = ou_idx  # 目标索引（自身）

            return map_tensor

        except Exception as e:
            print(f"❌ 构建单位映射矩阵失败 {layer_name}: {e}")
            return None

    def _build_identity_multiplier(self, layer_name: str) -> Optional[np.ndarray]:
        """
        构建单位倍数矩阵（回退模式）

        Args:
            layer_name: 层名称

        Returns:
            单位倍数矩阵
        """
        try:
            # 获取层的基本信息
            layer_idx = self._get_layer_index(layer_name)
            if layer_idx is None:
                return None

            # 使用配置中的层信息
            if hasattr(self, 'layer_config') and 'weight_names' in self.layer_config:
                if layer_name in self.layer_config['weight_names']:
                    idx = self.layer_config['weight_names'].index(layer_name)
                    layer_type = self.layer_config['layer_types'][idx]

                    if layer_type == 'conv':
                        in_ch, out_ch = 64, 64
                    elif layer_type == 'fc':
                        in_ch, out_ch = 512, 10
                    else:
                        in_ch, out_ch = 64, 64
                else:
                    in_ch, out_ch = 64, 64
            else:
                in_ch, out_ch = 64, 64

            # 构建单位倍数矩阵
            return np.ones((out_ch, in_ch), dtype=np.float32)

        except Exception as e:
            print(f"❌ 构建单位倍数矩阵失败 {layer_name}: {e}")
            return None

    def _get_layer_index(self, layer_name: str) -> Optional[int]:
        """获取层索引"""
        if hasattr(self, 'layer_config') and 'weight_names' in self.layer_config:
            try:
                return self.layer_config['weight_names'].index(layer_name)
            except ValueError:
                return None
        return None

    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况

        Returns:
            内存使用情况字典 (MB)
        """
        usage = {}

        if self.use_compressed and self.compressed_file:
            # 压缩文件的磁盘大小
            file_size_mb = Path(self.compressed_file).stat().st_size / (1024 * 1024)
            usage['compressed_file'] = file_size_mb
            usage['estimated_memory'] = file_size_mb * 1.2  # 估算内存使用（加上一些开销）
        else:
            # 回退模式的内存使用
            usage['fallback_mode'] = 1.0  # 很小的内存占用

        return usage

    def print_compression_stats(self):
        """打印压缩统计信息"""
        print("\n📊 压缩模式数据加载器统计:")

        if self.use_compressed and self.compressed_loader:
            print(f"  ✅ 使用压缩映射表")
            print(f"  📁 文件: {self.compressed_file}")

            if self.compressed_loader.header:
                total_mappings = self.compressed_loader.header['total_mappings']
                num_layers = self.compressed_loader.header['num_layers']

                print(f"  📈 层数: {num_layers}")
                print(f"  🔗 总映射: {total_mappings:,}")

                usage = self.get_memory_usage()
                if 'estimated_memory' in usage:
                    print(f"  💾 估算内存: {usage['estimated_memory']:.1f} MB")
        else:
            print(f"  ⚠️ 使用回退模式（轻量级）")
            print(f"  💾 内存占用: <1 MB")


def create_compressed_loader(model_name: str = 'Res50',
                           translate_name: str = 'ft_group_cluster_translate',
                           data_dir: str = './') -> CompressedPatternDataLoader:
    """
    创建压缩模式数据加载器的便捷函数

    Args:
        model_name: 模型名称
        translate_name: 转换方法名称
        data_dir: 数据目录

    Returns:
        压缩模式数据加载器实例
    """
    loader = CompressedPatternDataLoader(model_name, translate_name, data_dir)

    if loader.load_all_data():
        loader.print_compression_stats()
        return loader
    else:
        raise RuntimeError("创建压缩数据加载器失败")


if __name__ == "__main__":
    # 测试压缩数据加载器
    print("=" * 80)
    print("🧪 测试压缩模式数据加载器")
    print("=" * 80)

    try:
        loader = create_compressed_loader()

        # 测试获取映射数据
        layer_names = loader.get_all_layer_names()
        if layer_names:
            test_layer = layer_names[0]

            print(f"\n🎯 测试获取 {test_layer} 的数据:")

            # 测试映射数据
            map_data = loader.get_layer_map(test_layer)
            if map_data is not None:
                print(f"  📊 映射数据形状: {map_data.shape}")
                print(f"  📊 有效映射数量: {np.sum(map_data[:, :, 1] >= 0)}")

            # 测试倍数数据
            mult_data = loader.get_layer_multiplier(test_layer)
            if mult_data is not None:
                print(f"  📊 倍数数据形状: {mult_data.shape}")
                print(f"  📊 非单位倍数数量: {np.sum(np.abs(mult_data - 1.0) > 1e-6)}")

        print("\n✅ 测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
