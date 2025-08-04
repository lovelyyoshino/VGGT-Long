#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY文件读取和可视化脚本
用于读取VGGT-Long输出的点云文件并进行可视化

使用方法:
python read_ply.py [PLY文件路径]

默认读取路径: ./exps/._output_images_/2025-07-25-13-22-24/pcd/combined_pcd.ply
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("警告: trimesh未安装，将使用简单的PLY读取方法")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("提示: open3d未安装，将使用matplotlib进行可视化")


def read_ply_simple(file_path):
    """
    简单的PLY文件读取函数（不依赖外部库）
    """
    points = []
    colors = []
    
    with open(file_path, 'r') as f:
        # 读取头部信息
        line = f.readline().strip()
        if line != 'ply':
            raise ValueError("不是有效的PLY文件")
        
        vertex_count = 0
        in_header = True
        has_color = False
        
        while in_header:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property') and ('red' in line or 'green' in line or 'blue' in line):
                has_color = True
            elif line == 'end_header':
                in_header = False
        
        # 读取顶点数据
        for _ in range(vertex_count):
            line = f.readline().strip()
            if line:
                values = line.split()
                # 读取坐标
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                points.append([x, y, z])
                
                # 读取颜色（如果存在）
                if has_color and len(values) >= 6:
                    r, g, b = float(values[3]), float(values[4]), float(values[5])
                    colors.append([r/255.0, g/255.0, b/255.0])
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    
    return points, colors


def read_ply_trimesh(file_path):
    """
    使用trimesh读取PLY文件
    """
    mesh = trimesh.load(file_path)
    
    if hasattr(mesh, 'vertices'):
        points = mesh.vertices
        colors = None
        
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        
        return points, colors
    else:
        raise ValueError("无法从PLY文件中读取顶点数据")


def visualize_with_matplotlib(points, colors=None, title="点云可视化"):
    """
    使用matplotlib进行3D可视化
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    if colors is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=colors, s=1, alpha=0.6)
    else:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    
    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 添加颜色条
    plt.colorbar(scatter, shrink=0.5, aspect=5)
    
    # 设置相等的纵横比
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()


def visualize_with_open3d(points, colors=None):
    """
    使用Open3D进行可视化
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY点云可视化", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def print_point_cloud_info(points, colors=None, file_path=""):
    """
    打印点云信息
    """
    print("\n" + "="*50)
    print(f"PLY文件信息: {os.path.basename(file_path)}")
    print("="*50)
    print(f"文件路径: {file_path}")
    print(f"点云总数: {len(points):,}")
    print(f"是否有颜色信息: {'是' if colors is not None else '否'}")
    
    if len(points) > 0:
        print(f"\n坐标范围:")
        print(f"  X: {points[:, 0].min():.3f} ~ {points[:, 0].max():.3f}")
        print(f"  Y: {points[:, 1].min():.3f} ~ {points[:, 1].max():.3f}")
        print(f"  Z: {points[:, 2].min():.3f} ~ {points[:, 2].max():.3f}")
        
        print(f"\n坐标均值:")
        print(f"  X: {points[:, 0].mean():.3f}")
        print(f"  Y: {points[:, 1].mean():.3f}")
        print(f"  Z: {points[:, 2].mean():.3f}")
        
        print(f"\n坐标标准差:")
        print(f"  X: {points[:, 0].std():.3f}")
        print(f"  Y: {points[:, 1].std():.3f}")
        print(f"  Z: {points[:, 2].std():.3f}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="读取和可视化PLY点云文件")
    parser.add_argument("ply_file", nargs='?', 
                       default="./exps/._keyframes_/2025-08-04-11-16-12/pcd/combined_pcd_merged.ply",
                       help="PLY文件路径 (默认: ./exps/._keyframes_/2025-08-04-09-30-15/pcd/combined_pcd_merged.ply)")
    parser.add_argument("--no-vis", action="store_true", 
                       help="只打印信息，不进行可视化")
    parser.add_argument("--engine", choices=['matplotlib', 'open3d', 'auto'], default='auto',
                       help="指定可视化引擎 (默认: auto)")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.ply_file):
        print(f"错误: 文件不存在 - {args.ply_file}")
        
        # 尝试查找可能的文件
        exps_dir = "./exps"
        if os.path.exists(exps_dir):
            print(f"\n在 {exps_dir} 目录下查找PLY文件...")
            for root, dirs, files in os.walk(exps_dir):
                for file in files:
                    if file.endswith('.ply'):
                        print(f"  找到: {os.path.join(root, file)}")
        
        sys.exit(1)
    
    print(f"正在读取PLY文件: {args.ply_file}")
    
    try:
        # 尝试读取PLY文件
        if HAS_TRIMESH:
            print("使用trimesh读取...")
            points, colors = read_ply_trimesh(args.ply_file)
        else:
            print("使用内置方法读取...")
            points, colors = read_ply_simple(args.ply_file)
        
        # 打印点云信息
        print_point_cloud_info(points, colors, args.ply_file)
        
        # 可视化
        if not args.no_vis:
            print("\n开始可视化...")
            
            if args.engine == 'open3d' and HAS_OPEN3D:
                visualize_with_open3d(points, colors)
            elif args.engine == 'matplotlib':
                visualize_with_matplotlib(points, colors)
            elif args.engine == 'auto':
                if HAS_OPEN3D:
                    print("使用Open3D进行可视化...")
                    visualize_with_open3d(points, colors)
                else:
                    print("使用matplotlib进行可视化...")
                    visualize_with_matplotlib(points, colors)
            else:
                print(f"错误: 无法使用指定的可视化引擎 '{args.engine}'")
                sys.exit(1)
    
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()