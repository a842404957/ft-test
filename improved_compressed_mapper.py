#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的压缩冗余映射表生成器
解决OU索引溢出和重复记录问题
"""

import pickle
import struct
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class ImprovedCompressedMapper:
    """改进的压缩冗余映射表生成器"""

    # 文件格式常量
    MAGIC_NUMBER = 0x4D415053  # "MAPS" in hex
    VERSION = 2  # 新版本

    def __init__(self, model_name: str = 'Res50'):
        """初始化改进的压缩器"""
        self.model_name = model_name
        self.layer_mappings = {}  # {layer_name: Set[Tuple[source, target, multiplier]]}
        self.layer_info = {}     # {layer_name: (in_channels, out_channels)}

    def load_original_data(self) -> bool:
        """加载并分析原始数据"""
        print(f"📂 加载 {self.model_name} 的原始数据...")

        try:
            # 加载映射信息
            map_file = f'model_{self.model_name}_shape_and_value_similar_map_information.pkl'
            with open(map_file, 'rb') as f:
                map_data = pickle.load(f)

            # 加载倍数关系
            mult_file = f'model_{self.model_name}_shape_and_value_multiple_relationship_information.pkl'
            with open(mult_file, 'rb') as f:
                mult_data = pickle.load(f)

            print(f"  ✓ 加载映射信息: {map_file}")
            print(f"  ✓ 加载倍数关系: {mult_file}")

            # 解析每层映射数据
            total_mappings = 0
            valid_layers = 0
            total_ous = 0

            for layer_name in map_data.keys():
                if layer_name not in mult_data:
                    continue

                map_tensor = map_data[layer_name]
                mult_tensor = mult_data[layer_name]

                if len(map_tensor.shape) >= 2:
                    in_channels, out_channels = map_tensor.shape[:2]
                    total_ous += in_channels * out_channels

                    # 使用Set去重
                    unique_mappings = set()

                    for out_ch in range(out_channels):
                        for in_ch in range(in_channels):
                            if len(map_tensor.shape) == 3:
                                source_idx, target_idx = map_tensor[in_ch, out_ch].tolist()
                            else:
                                source_idx = map_tensor[in_ch, out_ch].item()
                                target_idx = map_tensor[in_ch, out_ch].item()

                            if target_idx >= 0:
                                # 使用层内唯一索引，避免全局溢出
                                local_source_ou = out_ch * in_channels + in_ch

                                # 计算倍数
                                if len(mult_tensor.shape) == 2:
                                    multiplier = mult_tensor[out_ch, in_ch].item()
                                elif len(mult_tensor.shape) == 4:
                                    multiplier = mult_tensor[out_ch, in_ch].mean().item()
                                elif len(mult_tensor.shape) == 3:
                                    multiplier = mult_tensor[out_ch, in_ch].mean().item()
                                else:
                                    multiplier = 1.0

                                unique_mappings.add((local_source_ou, target_idx, float(multiplier)))

                    if unique_mappings:
                        self.layer_mappings[layer_name] = unique_mappings
                        self.layer_info[layer_name] = (in_channels, out_channels)
                        total_mappings += len(unique_mappings)
                        valid_layers += 1

                        print(f"  ✓ {layer_name}: {len(unique_mappings)} 唯一映射 "
                              f"(去重前可能更多)")

            print(f"\n📊 数据解析完成:")
            print(f"  有效层数: {valid_layers}")
            print(f"  总OU数量: {total_ous:,}")
            print(f"  唯一映射数: {total_mappings:,}")

            return total_mappings > 0

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def compress_to_binary(self, output_file: str = None) -> bool:
        """压缩到改进的二进制格式"""
        if not self.layer_mappings:
            print("❌ 没有数据可压缩")
            return False

        if output_file is None:
            output_file = f'{self.model_name}_improved_redundancy_map.bin'

        print(f"🗜️ 压缩数据到: {output_file}")

        try:
            with open(output_file, 'wb') as f:
                # 1. 写入文件头
                self._write_header(f)

                # 2. 写入层索引表
                self._write_layer_index(f)

                # 3. 写入映射数据
                self._write_mapping_data(f)

            # 验证文件大小
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✅ 压缩完成: {file_size_mb:.1f} MB")

            return True

        except Exception as e:
            print(f"❌ 压缩失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _write_header(self, f):
        """写入改进的文件头"""
        # 文件标识和版本
        f.write(struct.pack('I', self.MAGIC_NUMBER))  # 4 bytes
        f.write(struct.pack('H', self.VERSION))        # 2 bytes

        # 层数和总映射数
        num_layers = len(self.layer_mappings)
        total_mappings = sum(len(mappings) for mappings in self.layer_mappings.values())

        f.write(struct.pack('H', num_layers))          # 2 bytes
        f.write(struct.pack('I', total_mappings))      # 4 bytes

        # 新增: 编码格式信息
        f.write(struct.pack('B', 1))  # 1 byte: 压缩格式版本
        f.write(struct.pack('B', 0))  # 1 byte: 保留

        # 预留字节
        f.write(b'\x00' * 50)                         # 50 bytes

        print(f"  📝 改进文件头: {num_layers} 层, {total_mappings:,} 映射")

    def _write_layer_index(self, f):
        """写入改进的层索引表"""
        layer_names = sorted(self.layer_mappings.keys())
        current_offset = 64 + len(layer_names) * 24  # header + layer index (每层24字节)

        for layer_name in layer_names:
            mappings = self.layer_mappings[layer_name]
            in_channels, out_channels = self.layer_info[layer_name]

            # 写入层索引条目 (改进为24字节)
            f.write(struct.pack('Q', current_offset))      # 8 bytes
            f.write(struct.pack('I', len(mappings)))      # 4 bytes
            f.write(struct.pack('H', in_channels))         # 2 bytes
            f.write(struct.pack('H', out_channels))        # 2 bytes
            f.write(struct.pack('I', in_channels * out_channels))  # 4 bytes: 总OU数
            f.write(struct.pack('B', 1))  # 1 byte: 层类型标记
            f.write(struct.pack('B', 0))  # 1 byte: 保留

            print(f"  📝 {layer_name}: {len(mappings)} 映射, "
                  f"总OU={in_channels*out_channels}, 偏移 {current_offset}")

            # 更新偏移量
            current_offset += len(mappings) * 8  # 每个映射8字节

    def _write_mapping_data(self, f):
        """写入改进的映射数据"""
        total_written = 0

        for layer_name in sorted(self.layer_mappings.keys()):
            mappings = sorted(self.layer_mappings[layer_name])  # 排序确保一致性

            for source_ou, target_ou, multiplier in mappings:
                # 改进格式: 8字节每个映射
                # source_ou: uint32 (4 bytes)
                # target_ou: uint32 (4 bytes)
                # multiplier: uint8 (1 byte) + 保留 (1 byte)

                # 确保是整数
                source_ou_int = int(source_ou)
                target_ou_int = int(target_ou)

                f.write(struct.pack('I', source_ou_int))      # 4 bytes
                f.write(struct.pack('I', target_ou_int))      # 4 bytes

                # 量化倍数到uint8 (0-255)
                if multiplier >= 0:
                    quantized_mult = min(255, int(multiplier * 255))
                else:
                    quantized_mult = 0  # 负倍数设为0

                f.write(struct.pack('B', quantized_mult))   # 1 byte
                f.write(struct.pack('B', 0))              # 1 byte 保留

                total_written += 1

                if total_written % 1000000 == 0:
                    print(f"    💾 已写入: {total_written:,} 映射")

        print(f"  💾 映射数据写入完成: {total_written:,} 条目")


class ImprovedCompressedLoader:
    """改进的压缩映射表加载器"""

    def __init__(self, file_path: str):
        """初始化加载器"""
        self.file_path = file_path
        self.header = {}
        self.layer_index = {}

    def load_header(self) -> bool:
        """加载文件头和层索引"""
        try:
            with open(self.file_path, 'rb') as f:
                # 读取文件头
                magic = struct.unpack('I', f.read(4))[0]
                version = struct.unpack('H', f.read(2))[0]
                num_layers = struct.unpack('H', f.read(2))[0]
                total_mappings = struct.unpack('I', f.read(4))[0]
                compression_format = struct.unpack('B', f.read(1))[0]

                # 验证文件格式
                if magic != ImprovedCompressedMapper.MAGIC_NUMBER:
                    print(f"❌ 无效的文件格式: {hex(magic)}")
                    return False

                if version != ImprovedCompressedMapper.VERSION:
                    print(f"❌ 不支持的版本: {version}")
                    return False

                self.header = {
                    'magic': magic,
                    'version': version,
                    'num_layers': num_layers,
                    'total_mappings': total_mappings,
                    'compression_format': compression_format
                }

                # 跳过预留字节
                f.read(52)

                # 读取层索引 (改进格式)
                self.layer_index = {}
                for i in range(num_layers):
                    offset = struct.unpack('Q', f.read(8))[0]
                    count = struct.unpack('I', f.read(4))[0]
                    in_channels = struct.unpack('H', f.read(2))[0]
                    out_channels = struct.unpack('H', f.read(2))[0]
                    total_ous = struct.unpack('I', f.read(4))[0]
                    layer_type = struct.unpack('B', f.read(1))[0]

                    layer_name = f'layer_{i}'
                    self.layer_index[layer_name] = {
                        'offset': offset,
                        'count': count,
                        'in_channels': in_channels,
                        'out_channels': out_channels,
                        'total_ous': total_ous,
                        'layer_type': layer_type
                    }

                print(f"✅ 加载改进文件头完成:")
                print(f"  层数: {num_layers}")
                print(f"  总映射: {total_mappings:,}")
                print(f"  压缩格式版本: {compression_format}")

                return True

        except Exception as e:
            print(f"❌ 加载文件头失败: {e}")
            return False

    def get_layer_mappings(self, layer_name: str) -> List[Tuple[int, int, float]]:
        """获取层的映射数据"""
        if layer_name not in self.layer_index:
            return []

        layer_info = self.layer_index[layer_name]
        mappings = []

        try:
            with open(self.file_path, 'rb') as f:
                f.seek(layer_info['offset'])

                for i in range(layer_info['count']):
                    source_ou = struct.unpack('I', f.read(4))[0]
                    target_ou = struct.unpack('I', f.read(4))[0]
                    quantized_mult = struct.unpack('B', f.read(1))[0]
                    _ = f.read(1)  # 跳过保留字节

                    # 反量化倍数
                    multiplier = quantized_mult / 255.0
                    mappings.append((source_ou, target_ou, multiplier))

        except Exception as e:
            print(f"❌ 读取层 {layer_name} 失败: {e}")

        return mappings

    def print_summary(self):
        """打印加载摘要"""
        if not self.header:
            print("❌ 文件头未加载")
            return

        print(f"\n📊 改进压缩冗余映射表摘要:")
        print(f"  文件: {self.file_path}")
        print(f"  版本: {self.header['version']}")
        print(f"  层数: {self.header['num_layers']}")
        print(f"  总映射: {self.header['total_mappings']:,}")

        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        print(f"  文件大小: {file_size_mb:.1f} MB")

        if self.header['total_mappings'] > 0:
            bytes_per_mapping = file_size_mb * 1024 * 1024 / self.header['total_mappings']
            print(f"  平均每映射: {bytes_per_mapping:.1f} 字节")


def main():
    """主函数"""
    print("=" * 80)
    print("🗜️ 改进压缩冗余映射表生成器")
    print("=" * 80)

    # 1. 创建改进压缩器
    compressor = ImprovedCompressedMapper('Res50')

    if compressor.load_original_data():
        output_file = 'Res50_improved_redundancy_map.bin'
        if compressor.compress_to_binary(output_file):
            print(f"\n✅ 改进压缩完成: {output_file}")
        else:
            print("❌ 改进压缩失败")
            return

    # 2. 测试改进加载器
    print(f"\n" + "=" * 80)
    print("📂 测试改进压缩文件加载")
    print("=" * 80)

    loader = ImprovedCompressedLoader(output_file)
    if loader.load_header():
        loader.print_summary()

        # 测试读取第一层
        first_layer = list(loader.layer_index.keys())[0]
        mappings = loader.get_layer_mappings(first_layer)
        print(f"\n🎯 测试读取 {first_layer}:")
        print(f"  映射数量: {len(mappings)}")
        if mappings:
            print(f"  示例映射: {mappings[:5]}")

    print("\n✅ 改进演示完成")


if __name__ == "__main__":
    main()