import argparse
import os
import sys
import torch
import gc
import trimesh
from vggt_long import VGGT_Long
from loop_utils.config_utils import load_config

def main():
    parser = argparse.ArgumentParser(description='VGGT-Long: 生成 glb 3D 模型并可视化')
    parser.add_argument('--image_dir', type=str, required=False, help='输入图片文件夹')
    parser.add_argument('--config', type=str, default='./configs/base_config.yaml', help='配置文件路径')
    parser.add_argument('--no_view', action='store_true', help='只生成 glb，不可视化')
    parser.add_argument('--glb_path', type=str, default=None, help='glb 文件保存路径（如已存在则直接可视化）')
    args = parser.parse_args()

    # 优先处理 glb_path 直接可视化
    if args.glb_path is not None:
        glb_path = args.glb_path
        if os.path.exists(glb_path):
            if not args.no_view:
                print(f"正在用 trimesh 可视化: {glb_path}")
                mesh_scene = trimesh.load(glb_path)
                mesh_scene.show()
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
    config = load_config(args.config)

    # 推理与点云生成
    vggt_long = VGGT_Long(image_dir, save_dir, config)
    vggt_long.run()

    # 导出 glb
    glb_path = args.glb_path or os.path.join(save_dir, "vggt_long_result.glb")
    vggt_long.export_glb(glb_path=glb_path)
    vggt_long.close()

    if not os.path.exists(glb_path):
        print(f"glb 文件未生成: {glb_path}，请检查上游流程是否成功。")
        return

    # 可视化 glb
    if not args.no_view:
        print(f"正在用 trimesh 可视化: {glb_path}")
        mesh_scene = trimesh.load(glb_path)
        mesh_scene.show()
    else:
        print(f"glb 文件已生成: {glb_path}")

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main() 