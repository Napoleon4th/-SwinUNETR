import os
import gzip
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import numpy as np


def reorganize_and_compress_brats_data(input_dir, output_dir=None, delete_original=False, verify_integrity=True):
    """
    重新组织并压缩BrATS21数据集（支持混合结构，处理kaggle下载的数据）

    支持两种结构：
    1. 嵌套结构：
        BraTS2021_00006/BraTS2021_00006_seg.nii/00000116_final_seg.nii
    2. 扁平结构：
        BraTS2021_00006/BraTS2021_00006_seg.nii
    转换为统一格式：
        BraTS2021_00006/BraTS2021_00006_seg.nii.gz

    参数:
        input_dir: 输入目录
        output_dir: 输出目录 (None表示在原目录处理)
        delete_original: 是否删除原始文件
        verify_integrity: 是否验证文件完整性

    返回:
        处理结果统计
    """
    input_dir = Path(input_dir)

    # 如果未指定输出目录，则在原目录处理
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        'total_cases': 0,
        'processed_files': 0,
        'skipped_files': 0,
        'errors': [],
        'file_mapping': {},
        'structure_types': {
            'nested': 0,  # 有子文件夹嵌套的
            'flat': 0,  # 直接的.nii文件
            'mixed': 0,  # 混合的
            'invalid': 0  # 无效的
        }
    }

    print(f"开始处理目录: {input_dir}")
    print("=" * 60)

    # 获取所有病例文件夹
    case_folders = []
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.startswith("BraTS2021_"):
            case_folders.append(item)

    stats['total_cases'] = len(case_folders)
    print(f"找到 {len(case_folders)} 个病例文件夹")

    # 处理每个病例
    for case_folder in tqdm(case_folders, desc="处理病例"):
        case_id = case_folder.name
        case_output_dir = output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # 扫描病例文件夹中的所有项目
        for item in case_folder.iterdir():
            process_item(item, case_output_dir, stats, delete_original, verify_integrity)

    return stats


def process_item(item, case_output_dir, stats, delete_original, verify_integrity):
    """
    处理单个项目（文件或文件夹）

    参数:
        item: 要处理的项目（Path对象）
        case_output_dir: 输出目录
        stats: 统计信息字典
        delete_original: 是否删除原始文件
        verify_integrity: 是否验证文件完整性
    """
    try:
        # 情况1: 如果是以.nii结尾的文件夹（嵌套结构）
        if item.is_dir() and item.name.endswith('.nii'):
            stats['structure_types']['nested'] += 1
            process_nested_nii_folder(item, case_output_dir, stats, delete_original, verify_integrity)

        # 情况2: 如果是直接的.nii文件（扁平结构）
        elif item.is_file() and item.name.endswith('.nii'):
            stats['structure_types']['flat'] += 1
            process_flat_nii_file(item, case_output_dir, stats, delete_original, verify_integrity)

        # 情况3: 如果是以.nii.gz结尾的文件（已压缩，跳过）
        elif item.is_file() and item.name.endswith('.nii.gz'):
            print(f"  ⏭️  已压缩，跳过: {item.name}")
            return

        # 情况4: 其他文件或文件夹
        elif item.is_dir():
            # 可能是深层结构，递归扫描
            for sub_item in item.iterdir():
                process_item(sub_item, case_output_dir, stats, delete_original, verify_integrity)
            stats['structure_types']['mixed'] += 1

    except Exception as e:
        error_msg = f"处理失败 {item}: {str(e)}"
        stats['errors'].append(error_msg)
        stats['skipped_files'] += 1
        print(f"\n⚠️  {error_msg}")


def process_nested_nii_folder(nii_folder, case_output_dir, stats, delete_original, verify_integrity):
    """
    处理嵌套的.nii文件夹结构

    参数:
        nii_folder: .nii文件夹路径
        case_output_dir: 输出目录
        stats: 统计信息字典
        delete_original: 是否删除原始文件
        verify_integrity: 是否验证文件完整性
    """
    # 目标文件名
    target_filename_base = nii_folder.name  # 例如BraTS2021_00006_seg.nii
    final_filename = target_filename_base + '.gz'  # 例如BraTS2021_00006_seg.nii.gz

    # 在.nii文件夹中查找.nii文件
    nii_files = []

    # 优先查找final文件
    nii_files = list(nii_folder.glob("*final*.nii"))

    if not nii_files:
        # 查找所有.nii文件
        nii_files = list(nii_folder.glob("*.nii"))

    if not nii_files:
        # 如果还没有，可能在更深层的子文件夹中
        for sub_item in nii_folder.rglob("*.nii"):
            nii_files.append(sub_item)

    if nii_files:
        # 取第一个.nii文件
        source_file = nii_files[0]
        target_file = case_output_dir / final_filename

        process_single_nii_file(
            source_file, target_file, nii_folder, stats,
            delete_original, verify_integrity, is_nested=True
        )
    else:
        stats['skipped_files'] += 1
        print(f"\n⚠️  文件夹中没有找到.nii文件: {nii_folder}")


def process_flat_nii_file(nii_file, case_output_dir, stats, delete_original, verify_integrity):
    """
    处理扁平的.nii文件

    参数:
        nii_file: .nii文件路径
        case_output_dir: 输出目录
        stats: 统计信息字典
        delete_original: 是否删除原始文件
        verify_integrity: 是否验证文件完整性
    """
    # 目标文件名
    target_filename = nii_file.name + '.gz'  # BraTS2021_00006_seg.nii.gz
    target_file = case_output_dir / target_filename

    process_single_nii_file(
        nii_file, target_file, nii_file.parent, stats,
        delete_original, verify_integrity, is_nested=False
    )


def process_single_nii_file(source_file, target_file, original_parent, stats,
                            delete_original, verify_integrity, is_nested=False):
    """
    处理单个.nii文件

    参数:
        source_file: 源文件路径
        target_file: 目标文件路径
        original_parent: 原始父目录（用于删除）
        stats: 统计信息字典
        delete_original: 是否删除原始文件
        verify_integrity: 是否验证文件完整性
        is_nested: 是否来自嵌套结构
    """
    try:
        # 检查目标文件是否已存在
        if target_file.exists():
            print(f"  ⏭️  目标文件已存在，跳过: {target_file.name}")
            return

        # 1. 验证文件完整性
        if verify_integrity:
            try:
                img = nib.load(str(source_file))
                data = img.get_fdata()
                stats['file_mapping'][str(source_file)] = {
                    'target': str(target_file),
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'affine': img.affine.tolist(),
                    'is_nested': is_nested
                }
            except Exception as e:
                stats['errors'].append(f"验证失败 {source_file}: {str(e)}")
                stats['skipped_files'] += 1
                return

        # 2. 压缩文件
        compress_nii_file(source_file, target_file)

        stats['processed_files'] += 1

        # 3. 可选择：删除原始文件
        if delete_original:
            # 删除源文件
            source_file.unlink()

            # 如果是嵌套结构，尝试删除父文件夹
            if is_nested:
                try:
                    original_parent.rmdir()  # 删除.nii文件夹
                except:
                    pass  # 文件夹非空，不删除

    except Exception as e:
        error_msg = f"处理失败 {source_file} -> {target_file}: {str(e)}"
        stats['errors'].append(error_msg)
        stats['skipped_files'] += 1
        print(f"\n⚠️  {error_msg}")


def compress_nii_file(source_path, target_path):
    """
    压缩NIfTI文件为.gz格式

    参数:
        source_path: 源文件路径
        target_path: 目标文件路径 (.nii.gz)
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    # 确保目标目录存在
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用gzip压缩
    with open(source_path, 'rb') as f_in:
        with gzip.open(target_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)  # 16MB chunks

    # 验证压缩文件
    verify_compressed_file(source_path, target_path)

    return target_path


def verify_compressed_file(original_path, compressed_path):
    """
    验证压缩文件

    参数:
        original_path: 原始文件路径
        compressed_path: 压缩文件路径
    """
    original_size = original_path.stat().st_size
    compressed_size = compressed_path.stat().st_size

    # 验证压缩文件可以正确加载
    try:
        img = nib.load(str(compressed_path))
        data = img.get_fdata()

        compression_ratio = compressed_size / original_size * 100 if original_size > 0 else 0

        print(f"  ✓ 压缩成功: {original_path.name}")
        print(f"     图像形状: {data.shape}")

        return True
    except Exception as e:
        print(f"  ✗ 验证失败: {compressed_path} - {str(e)}")
        # 删除无效的压缩文件
        compressed_path.unlink(missing_ok=True)
        raise e


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def analyze_directory_structure(input_dir):
    """
    分析目录结构，识别不同类型的文件组织方式

    参数:
        input_dir: 输入目录
    """
    input_dir = Path(input_dir)

    print("\n🔍 分析目录结构...")
    print("=" * 60)

    structure_stats = {
        'nested_folders': [],  # 有.nii文件夹的
        'flat_files': [],  # 有直接.nii文件的
        'mixed_cases': [],  # 混合结构的
        'compressed_files': [],  # 已有.gz文件的
        'other_files': []  # 其他文件
    }

    # 获取所有病例文件夹
    case_folders = []
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.startswith("BraTS2021_"):
            case_folders.append(item)

    print(f"找到 {len(case_folders)} 个病例文件夹")

    # 分析每个病例
    for case_folder in case_folders[:5]:
        print(f"\n分析: {case_folder.name}")

        nested_items = []
        flat_items = []
        compressed_items = []

        for item in case_folder.iterdir():
            if item.is_dir() and item.name.endswith('.nii'):
                nested_items.append(item.name)
            elif item.is_file() and item.name.endswith('.nii'):
                flat_items.append(item.name)
            elif item.is_file() and item.name.endswith('.nii.gz'):
                compressed_items.append(item.name)

        if nested_items and flat_items:
            structure_stats['mixed_cases'].append(case_folder.name)
            print(f"  ⚠️  混合结构: {len(nested_items)}个文件夹 + {len(flat_items)}个文件")
        elif nested_items:
            structure_stats['nested_folders'].append(case_folder.name)
            print(f"  📁 嵌套结构: {len(nested_items)}个文件夹")
        elif flat_items:
            structure_stats['flat_files'].append(case_folder.name)
            print(f"  📄 扁平结构: {len(flat_items)}个文件")

        if compressed_items:
            print(f"  ⏭️  已压缩: {len(compressed_items)}个.gz文件")

    # 打印统计
    print("\n" + "=" * 60)
    print("📊 结构分析统计:")
    print(f"  嵌套结构病例: {len(structure_stats['nested_folders'])}")
    print(f"  扁平结构病例: {len(structure_stats['flat_files'])}")
    print(f"  混合结构病例: {len(structure_stats['mixed_cases'])}")
    print(f"  已有压缩文件: {len(structure_stats['compressed_files'])}")

    return structure_stats


def save_statistics(stats, output_dir):
    """保存处理统计信息"""
    output_dir = Path(output_dir)

    # 保存统计在json文件
    stats_path = output_dir / "processing_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_stats = stats.copy()
        json_stats['file_mapping'] = {
            k: v for k, v in json_stats['file_mapping'].items()
        }
        json.dump(json_stats, f, indent=2, ensure_ascii=False)

    # 保存处理报告
    report_path = output_dir / "processing_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BrATS数据处理报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"处理时间: {stats.get('timestamp', 'N/A')}\n")
        f.write(f"输入目录: {stats.get('input_dir', 'N/A')}\n")
        f.write(f"输出目录: {stats.get('output_dir', 'N/A')}\n\n")

        f.write("-" * 60 + "\n")
        f.write("处理统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"总病例数: {stats['total_cases']}\n")
        f.write(f"成功处理: {stats['processed_files']}\n")
        f.write(f"跳过文件: {stats['skipped_files']}\n")
        f.write(f"错误数量: {len(stats['errors'])}\n\n")

        f.write("-" * 60 + "\n")
        f.write("结构类型统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"嵌套结构: {stats['structure_types']['nested']}\n")
        f.write(f"扁平结构: {stats['structure_types']['flat']}\n")
        f.write(f"混合结构: {stats['structure_types']['mixed']}\n")
        f.write(f"无效结构: {stats['structure_types']['invalid']}\n\n")

        if stats['errors']:
            f.write("-" * 60 + "\n")
            f.write("错误列表\n")
            f.write("-" * 60 + "\n")
            for i, error in enumerate(stats['errors'], 1):
                f.write(f"{i:3d}. {error}\n")

    return stats_path, report_path


def smart_reorganize_brats_data(input_dir, output_dir=None, delete_original=False, verify_integrity=True):
    """
    重新组织BrATS数据（主函数）

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        delete_original: 是否删除原始文件
        verify_integrity: 是否验证文件完整性
    """
    print("=" * 60)
    print("BrATS数据重组工具")
    print("=" * 60)

    # 1. 先分析目录结构
    structure_stats = analyze_directory_structure(input_dir)

    # 2. 询问用户确认
    print("\n将统一转换为:")
    print("   BraTS2021_XXXXX/BraTS2021_XXXXX_xxx.nii.gz")

    response = input("\n是否继续处理? (y/n): ").strip().lower()
    if response != 'y':
        print("操作已取消")
        return None

    # 3. 执行处理
    stats = reorganize_and_compress_brats_data(
        input_dir=input_dir,
        output_dir=output_dir,
        delete_original=delete_original,
        verify_integrity=verify_integrity
    )

    return stats


def main():
    """主函数"""
    import time
    from datetime import datetime

    # 配置参数
    input_directory = r"~" # 数据文件夹，本人电脑中为 "E:\数据集\Data\TrainingData"，即嵌套，扁平结构的上一级文件夹
    output_directory = None  # None表示在原目录处理
    delete_original = False
    verify_integrity = True  # 验证文件完整性

    print("BrATS数据重组和压缩工具")
    print("=" * 60)
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory if output_directory else '原目录'}")
    print(f"删除原始文件: {'是' if delete_original else '否'}")
    print(f"验证文件完整性: {'是' if verify_integrity else '否'}")
    print("=" * 60)

    # 确认操作
    if delete_original:
        confirmation = input("\n⚠️  警告：这将删除原始文件！确认继续？(yes/no): ")
        if confirmation.lower() != 'yes':
            print("操作已取消")
            return

    # 记录开始时间
    start_time = time.time()

    try:
        # 智能处理
        stats = smart_reorganize_brats_data(
            input_dir=input_directory,
            output_dir=output_directory,
            delete_original=delete_original,
            verify_integrity=verify_integrity
        )

        if stats is None:
            return 0

        # 添加元数据
        stats['timestamp'] = datetime.now().isoformat()
        stats['input_dir'] = str(input_directory)
        stats['output_dir'] = str(output_directory if output_directory else input_directory)

        # 计算耗时
        elapsed_time = time.time() - start_time
        stats['elapsed_time'] = elapsed_time

        # 保存统计信息
        output_dir = Path(output_directory if output_directory else input_directory)
        stats_path, report_path = save_statistics(stats, output_dir)

        # 打印报告
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"总病例数: {stats['total_cases']}")
        print(f"成功处理: {stats['processed_files']} 个文件")
        print(f"跳过文件: {stats['skipped_files']}")
        print(f"错误数量: {len(stats['errors'])}")

        print("\n结构类型统计:")
        print(f"  嵌套结构: {stats['structure_types']['nested']}")
        print(f"  扁平结构: {stats['structure_types']['flat']}")
        print(f"  混合结构: {stats['structure_types']['mixed']}")

        if stats['errors']:
            print(f"\n有 {len(stats['errors'])} 个错误:")
            for i, error in enumerate(stats['errors'][:5], 1):  # 显示前5个错误
                print(f"  {i}. {error}")
            if len(stats['errors']) > 5:
                print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")

        print(f"\n统计信息已保存到: {stats_path}")
        print(f"处理报告已保存到: {report_path}")

        # 显示一些转换示例
        if stats['file_mapping']:
            print(f"\n转换示例:")
            examples = list(stats['file_mapping'].items())[:3]
            for i, (src, info) in enumerate(examples, 1):
                src_name = Path(src).name
                target_name = Path(info['target']).name
                structure_type = "嵌套" if info.get('is_nested', False) else "扁平"
                print(f"  {i}. [{structure_type}] {src_name} -> {target_name}")
                print(f"     形状: {info['shape']}")

        # 提供清理建议
        if not delete_original and stats['processed_files'] > 0:
            print("\n清理建议:")
            print(f"  已处理 {stats['processed_files']} 个文件，原始文件仍保留")
            print(f"  如需清理，可运行: python script.py --input '{input_directory}' --delete")

    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def quick_convert_mode():
    """快速转换（不验证，直接压缩）"""
    input_directory = r"E:\数据集\Data\TrainingData"

    print("快速转换模式")
    print("注意：此模式不验证文件完整性，直接压缩")
    print(f"处理目录: {input_directory}")

    stats = {
        'processed': 0,
        'skipped': 0,
        'errors': []
    }

    for root, dirs, files in os.walk(input_directory):
        for item_name in dirs + files:
            item_path = Path(root) / item_name

            # 处理.nii文件夹
            if item_path.is_dir() and item_name.endswith('.nii'):
                process_nii_folder_quick(item_path, stats)

            # 处理.nii文件
            elif item_path.is_file() and item_name.endswith('.nii'):
                process_nii_file_quick(item_path, stats)

    print(f"\n快速转换完成!")
    print(f"  成功处理: {stats['processed']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"  错误: {len(stats['errors'])}")


def process_nii_folder_quick(nii_folder, stats):
    """快速处理.nii文件夹"""
    # 查找.nii文件
    nii_files = list(nii_folder.rglob("*.nii"))
    if nii_files:
        source_file = nii_files[0]
        target_file = nii_folder.parent / (nii_folder.name + ".gz")

        try:
            with open(source_file, 'rb') as f_in:
                with gzip.open(target_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            stats['processed'] += 1
            print(f"✓ 嵌套: {source_file.name} -> {target_file.name}")

        except Exception as e:
            stats['errors'].append(str(e))


def process_nii_file_quick(nii_file, stats):
    """快速处理.nii文件"""
    target_file = nii_file.parent / (nii_file.name + ".gz")

    if target_file.exists():
        stats['skipped'] += 1
        return

    try:
        with open(nii_file, 'rb') as f_in:
            with gzip.open(target_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        stats['processed'] += 1
        print(f"✓ 扁平: {nii_file.name} -> {target_file.name}")

    except Exception as e:
        stats['errors'].append(str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='智能重组和压缩BrATS数据')
    parser.add_argument('--input', '-i', default=r"E:\数据集\Data\TrainingData",
                        help='输入目录路径')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录路径（默认：原目录）')
    parser.add_argument('--delete', '-d', action='store_true',
                        help='删除原始文件')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='快速模式（不验证）')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='只分析目录结构，不处理')

    args = parser.parse_args()

    if args.analyze:
        analyze_directory_structure(args.input)
    elif args.quick:
        quick_convert_mode()
    else:
        # 运行主程序
        main()