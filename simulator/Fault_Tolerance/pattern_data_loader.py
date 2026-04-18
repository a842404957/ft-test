#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重模式数据加载器
读取并解析由 main.py 生成的权重模式复用数据
"""

import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class PatternDataLoader:
    """权重模式数据加载器"""
    
    def __init__(self, model_name: str = 'Vgg16', 
                 translate_name: str = 'ft_group_cluster_translate',
                 data_dir: str = './'):
        """
        初始化数据加载器
        
        Args:
            model_name: 模型名称 (Vgg16, Res18, Res50, WRN)
            translate_name: 转换方法名称
            data_dir: 数据文件目录
        """
        self.model_name = model_name
        self.translate_name = translate_name
        self.data_dir = Path(data_dir)
        
        # 存储加载的数据
        self.group_information = None
        self.map_information = None
        self.multiple_relationship_information = None
        self.reuse_ratio_information = None
        self.pattern_mask = None
        self.layer_config = None
        
        print(f"🔍 初始化权重模式数据加载器")
        print(f"  模型: {model_name}")
        print(f"  转换方法: {translate_name}")
        
    def load_all_data(self) -> bool:
        """
        加载所有相关数据文件
        
        Returns:
            bool: 是否成功加载
        """
        try:
            print("📂 开始加载权重模式数据...")

            # 0. 显式冗余组信息（FT主路径）
            self.group_information = self._load_pkl_file(
                f'model_{self.model_name}_{self.translate_name}_group_information.pkl'
            )
            
            # 1. 加载映射信息 (核心数据)
            self.map_information = self._load_pkl_file(
                f'model_{self.model_name}_{self.translate_name}_map_information.pkl',
                alternative_name=f'model_{self.model_name}_shape_and_value_similar_map_information.pkl'
            )
            
            # 2. 加载倍数关系信息
            self.multiple_relationship_information = self._load_pkl_file(
                f'model_{self.model_name}_{self.translate_name}_multiple_relationship_information.pkl',
                alternative_name=f'model_{self.model_name}_shape_and_value_multiple_relationship_information.pkl'
            )
            
            # 3. 加载重用率信息
            self.reuse_ratio_information = self._load_pkl_file(
                f'model_{self.model_name}_{self.translate_name}_reuse_ratio_information.pkl',
                alternative_name=f'model_{self.model_name}_shape_and_value_reuse_ratio_information.pkl'
            )
            
            # 4. 加载模式掩码
            self.pattern_mask = self._load_pkl_file(
                f'model_{self.model_name}_{self.translate_name}_mask.pkl',
                alternative_name=f'model_{self.model_name}_pattern_mask.pkl'
            )
            
            # 5. 构建层配置信息
            self._build_layer_config()
            
            # 验证数据
            if self._validate_data():
                print("✅ 所有数据加载成功")
                self._print_data_summary()
                return True
            else:
                print("❌ 数据验证失败")
                return False
                
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_pkl_file(self, filename: str, alternative_name: str = None) -> Any:
        """
        加载 pkl 文件
        
        Args:
            filename: 主文件名
            alternative_name: 备选文件名
            
        Returns:
            加载的数据
        """
        # 尝试主文件名
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"  ✓ 加载: {filename}")
            return data
        
        # 尝试备选文件名
        if alternative_name:
            alt_filepath = self.data_dir / alternative_name
            if alt_filepath.exists():
                with open(alt_filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"  ✓ 加载: {alternative_name}")
                return data
        
        print(f"  ⚠️ 未找到文件: {filename}")
        return None
    
    def _build_layer_config(self):
        """构建层配置信息"""
        # 根据模型名称获取层配置
        if self.model_name == 'Vgg16':
            self.layer_config = {
                'weight_names': [
                    'conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight',
                    'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight',
                    'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                    'conv13.weight', 'fc1.weight', 'fc2.weight', 'fc3.weight'
                ],
                'layer_types': ['conv'] * 13 + ['fc'] * 3,
                'num_layers': 16
            }
        elif self.model_name == 'Res18':
            self.layer_config = {
                'weight_names': [f'conv{i}.weight' for i in range(1, 18)] + 
                               [f'shortcut{i}.weight' for i in range(1, 4)] + 
                               ['fc.weight'],
                'layer_types': ['conv'] * 20 + ['fc'],
                'num_layers': 21
            }
        else:
            # 从group_information或map_information推断
            if self.group_information:
                self.layer_config = {
                    'weight_names': list(self.group_information.keys()),
                    'num_layers': len(self.group_information)
                }
            elif self.map_information:
                self.layer_config = {
                    'weight_names': list(self.map_information.keys()),
                    'num_layers': len(self.map_information)
                }
            else:
                self.layer_config = None
    
    def _validate_data(self) -> bool:
        """验证加载的数据完整性"""
        if self.map_information is None and self.group_information is None:
            print("  ❌ 缺少映射信息/分组信息（至少需要一种核心数据）")
            return False
        
        # 检查数据一致性
        if self.map_information and self.multiple_relationship_information:
            if len(self.map_information) != len(self.multiple_relationship_information):
                print("  ⚠️ 映射信息和倍数关系信息层数不一致")
        
        if self.map_information and self.reuse_ratio_information:
            if len(self.map_information) != len(self.reuse_ratio_information):
                print("  ⚠️ 映射信息和重用率信息层数不一致")
        
        return True
    
    def _print_data_summary(self):
        """打印数据摘要"""
        print("\n" + "=" * 60)
        print("数据加载摘要")
        print("=" * 60)

        if self.group_information:
            print(f"✓ 分组信息: {len(self.group_information)} 层")
            for layer_name, group_data in list(self.group_information.items())[:3]:
                group_count = group_data.get('group_count', len(group_data.get('groups', []))) if isinstance(group_data, dict) else 0
                print(f"  - {layer_name}: {group_count} groups")
        
        if self.map_information:
            print(f"✓ 映射信息: {len(self.map_information)} 层")
            for layer_name, map_data in list(self.map_information.items())[:3]:
                print(f"  - {layer_name}: {map_data.shape}")
        
        if self.multiple_relationship_information:
            print(f"✓ 倍数关系信息: {len(self.multiple_relationship_information)} 层")
        
        if self.reuse_ratio_information:
            print(f"✓ 重用率信息: {len(self.reuse_ratio_information)} 层")
            # 处理可能是tensor或float的情况
            total_reuse = sum(
                ratio.item() if torch.is_tensor(ratio) else ratio 
                for ratio in self.reuse_ratio_information.values()
            )
            avg_reuse = total_reuse / len(self.reuse_ratio_information)
            print(f"  平均重用率: {avg_reuse:.4f}")
        
        if self.pattern_mask:
            print(f"✓ 模式掩码: {len(self.pattern_mask)} 层")
        
        print("=" * 60 + "\n")
    
    def get_layer_map(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        获取指定层的映射信息
        
        Args:
            layer_name: 层名称
            
        Returns:
            映射矩阵 (in_channels, out_channels, 2)
        """
        if self.map_information and layer_name in self.map_information:
            return self.map_information[layer_name]
        return None

    def get_layer_group_info(self, layer_name: str) -> Optional[Dict]:
        """获取指定层的显式冗余组信息"""
        if self.group_information and layer_name in self.group_information:
            return self.group_information[layer_name]
        return None
    
    def get_layer_multiplier(self, layer_name: str) -> Optional[torch.Tensor]:
        """获取指定层的倍数关系"""
        if self.multiple_relationship_information and layer_name in self.multiple_relationship_information:
            return self.multiple_relationship_information[layer_name]
        return None
    
    def get_layer_reuse_ratio(self, layer_name: str) -> float:
        """获取指定层的重用率"""
        if self.reuse_ratio_information and layer_name in self.reuse_ratio_information:
            ratio = self.reuse_ratio_information[layer_name]
            return ratio.item() if torch.is_tensor(ratio) else ratio
        return 0.0
    
    def get_layer_mask(self, layer_name: str) -> Optional[torch.Tensor]:
        """获取指定层的模式掩码"""
        if self.pattern_mask and layer_name in self.pattern_mask:
            return self.pattern_mask[layer_name]
        return None
    
    def get_all_layer_names(self) -> List[str]:
        """获取所有层名称"""
        if self.group_information:
            return list(self.group_information.keys())
        elif self.map_information:
            return list(self.map_information.keys())
        elif self.layer_config:
            return self.layer_config['weight_names']
        return []
    
    def export_summary(self, output_file: str = 'pattern_data_summary.json'):
        """导出数据摘要到文件"""
        import json
        
        summary = {
            'model_name': self.model_name,
            'translate_name': self.translate_name,
            'num_layers': len(self.group_information) if self.group_information else (len(self.map_information) if self.map_information else 0),
            'layers': {}
        }
        
        if self.group_information or self.map_information:
            for layer_name in self.get_all_layer_names():
                layer_info = {
                    'map_shape': list(self.get_layer_map(layer_name).shape) if self.get_layer_map(layer_name) is not None else None,
                    'reuse_ratio': self.get_layer_reuse_ratio(layer_name),
                }
                if self.get_layer_group_info(layer_name) is not None:
                    layer_info['group_count'] = self.get_layer_group_info(layer_name).get('group_count', 0)
                
                if self.get_layer_mask(layer_name) is not None:
                    layer_info['mask_shape'] = list(self.get_layer_mask(layer_name).shape)
                
                summary['layers'][layer_name] = layer_info
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据摘要已导出到: {output_file}")


def test_loader():
    """测试数据加载器"""
    print("🧪 测试权重模式数据加载器\n")
    
    loader = PatternDataLoader(
        model_name='Vgg16',
        translate_name='ft_group_cluster_translate',
        data_dir='./'
    )
    
    if loader.load_all_data():
        print("\n✅ 数据加载测试通过")
        
        # 测试查询接口
        layer_names = loader.get_all_layer_names()
        if layer_names:
            test_layer = layer_names[0]
            print(f"\n🔍 测试层查询: {test_layer}")
            print(f"  映射矩阵形状: {loader.get_layer_map(test_layer).shape}")
            print(f"  重用率: {loader.get_layer_reuse_ratio(test_layer):.4f}")
        
        # 导出摘要
        loader.export_summary()
    else:
        print("\n❌ 数据加载测试失败")


if __name__ == "__main__":
    test_loader()
