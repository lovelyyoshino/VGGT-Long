import argparse
import os
import sys
import torch
import gc
import trimesh
import time
from datetime import datetime, timedelta
from vggt_long import VGGT_Long
from loop_utils.config_utils import load_config
from loop_utils.sim3utils import merge_ply_files

# python export_glb.py --image_dir ./output_images/ --config ./configs/base_config.yaml --merge_all

def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))

def main():
    # 总开始时间
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description='VGGT-Long: 生成 glb 3D 模型并可视化')
    parser.add_argument('--image_dir', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--config', type=str, default='./configs/base_config.yaml', help='配置文件路径')
    parser.add_argument('--no_view', action='store_true', help='只生成 glb，不可视化')
    parser.add_argument('--glb_path', type=str, default=None, help='glb 文件保存路径（如已存在则直接可视化）')
    parser.add_argument('--merge_all', action='store_true', help='合并所有点云数据生成GLB')
    args = parser.parse_args()

    # 优先处理 glb_path 直接可视化
    if args.glb_path is not None:
        glb_path = args.glb_path
        if os.path.exists(glb_path):
            if not args.no_view:
                print(f"正在用 trimesh 可视化: {glb_path}")
                viz_start_time = time.time()
                mesh_scene = trimesh.load(glb_path)
                mesh_scene.show()
                viz_time = time.time() - viz_start_time
                print(f"可视化耗时: {format_time(viz_time)}")
            else:
                print(f"glb 文件已存在: {glb_path}")
            return
        else:
            print(f"指定的 glb 文件不存在: {glb_path}，将尝试生成 glb 文件。")

    # 未指定 glb_path 或文件不存在，需 image_dir
    if not args.image_dir:
        print('未指定 --image_dir，且 --glb_path 不存在或无效，无法继续。')
        sys.exit(1)

    # 设置输出目录
    image_dir = args.image_dir
    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'
    save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'实验目录: {save_dir}')

    # 加载配置
    print("加载配置文件...")
    config_start_time = time.time()
    config = load_config(args.config)
    config_time = time.time() - config_start_time
    print(f"配置加载耗时: {format_time(config_time)}")

    # 推理与点云生成
    print("开始 VGGT-Long 推理与点云生成...")
    vggt_start_time = time.time()
    vggt_long = VGGT_Long(image_dir, save_dir, config)
    init_time = time.time() - vggt_start_time
    print(f"VGGT-Long 初始化耗时: {format_time(init_time)}")
    
    run_start_time = time.time()
    vggt_long.run()
    run_time = time.time() - run_start_time
    print(f"VGGT-Long 运行耗时: {format_time(run_time)}")

    # 导出 glb
    print("开始导出 GLB 文件...")
    export_start_time = time.time()
    
    # 根据参数选择导出方式
    if args.merge_all:
        print("使用合并模式导出所有点云数据...")
        merged_glb_path = args.glb_path or os.path.join(save_dir, "vggt_long_merged_result.glb")
        vggt_long.export_merged_glb(glb_path=merged_glb_path)
        glb_path = merged_glb_path
        
        # 合并模式下也生成合并的PLY文件
        print("合并模式：同时生成合并的PLY文件...")
        ply_merge_start = time.time()
        all_ply_path = os.path.join(save_dir, 'pcd/combined_pcd_merged.ply')
        input_dir = os.path.join(save_dir, 'pcd')
        print("正在合并所有点云文件...")
        merge_ply_files(input_dir, all_ply_path)
        print(f'合并PLY文件已保存: {all_ply_path}')
        ply_merge_time = time.time() - ply_merge_start
        print(f"PLY合并耗时: {format_time(ply_merge_time)}")
        
    else:
        print("使用默认模式导出最后一个块的数据...")
        glb_path = args.glb_path or os.path.join(save_dir, "vggt_long_result.glb")
        vggt_long.export_glb(glb_path=glb_path)
    
    export_time = time.time() - export_start_time
    print(f"GLB 导出耗时: {format_time(export_time)}")
    
    # 清理资源
    cleanup_start_time = time.time()
    vggt_long.close()
    cleanup_time = time.time() - cleanup_start_time
    print(f"资源清理耗时: {format_time(cleanup_time)}")

    if not os.path.exists(glb_path):
        print(f"glb 文件未生成: {glb_path}，请检查上游流程是否成功。")
        return

    # 可视化 glb
    if not args.no_view:
        print(f"正在用 trimesh 可视化: {glb_path}")
        viz_start_time = time.time()
        mesh_scene = trimesh.load(glb_path)
        mesh_scene.show()
        viz_time = time.time() - viz_start_time
        print(f"可视化耗时: {format_time(viz_time)}")
    else:
        print(f"glb 文件已生成: {glb_path}")

    # 最终清理
    final_cleanup_start = time.time()
    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()
    final_cleanup_time = time.time() - final_cleanup_start
    print(f"最终清理耗时: {format_time(final_cleanup_time)}")
    
    # 总时间统计
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"时间统计报告:")
    print(f"{'='*50}")
    print(f"配置加载:     {format_time(config_time)}")
    print(f"VGGT初始化:   {format_time(init_time)}")
    print(f"VGGT运行:     {format_time(run_time)}")
    print(f"GLB导出:      {format_time(export_time)}")
    if args.merge_all:
        print(f"PLY合并:      {format_time(ply_merge_time)}")
    print(f"资源清理:     {format_time(cleanup_time)}")
    if not args.no_view:
        print(f"可视化:       {format_time(viz_time)}")
    print(f"最终清理:     {format_time(final_cleanup_time)}")
    print(f"{'='*50}")
    print(f"总耗时:       {format_time(total_time)}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main() 