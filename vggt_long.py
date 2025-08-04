"""
VGGT-Long: 用于长序列图像的视觉几何全局变换器
本文件实现了一个用于处理长序列图像的3D重建和SLAM系统，支持循环检测和优化。

主要功能：
1. 将长序列图像分块处理以节省内存
2. 使用VGGT模型进行深度估计和位姿预测
3. 支持DBoW2和DNIO v2两种循环检测方法
4. 使用Sim3变换进行块间对齐
5. 循环闭合优化以减少累积误差
6. 生成点云和相机轨迹可视化
"""

import numpy as np
import argparse
import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2
import gc

# 尝试导入ONNX运行时，用于天空分割（可选功能）
try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

# 导入循环检测相关模块
from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW

# 导入VGGT模型相关模块
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 导入Sim3优化相关模块
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

# 图像处理和可视化
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import sys

# 配置文件加载
from loop_utils.config_utils import load_config
from loop_utils.visual_util import predictions_to_glb

def remove_duplicates(data_list):
    """
    移除重复的循环检测对
    
    Args:
        data_list: 循环检测结果列表，格式为[(frame1, range1, frame2, range2), ...]
                  例如：[(67, (3386, 3406), 48, (2435, 2455)), ...]
    
    Returns:
        list: 去重后的循环检测结果列表
        
    功能说明：
        - 移除自环（frame1 == frame2的情况）
        - 移除重复的帧对（相同的frame1和frame2组合）
    """
    seen = {}  # 用于记录已经处理过的帧对
    result = []
    
    for item in data_list:
        # 跳过自环（同一帧与自己形成的循环）
        if item[0] == item[2]:
            continue

        # 创建唯一键，用于识别重复的帧对
        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result

class LongSeqResult:
    """
    用于存储长序列处理结果的数据类
    
    存储所有块处理后合并的结果，包括：
    - 相机参数（内参、外参）
    - 深度图和置信度
    - 世界坐标系下的点云数据
    - 相机位姿轨迹
    """
    def __init__(self):
        self.combined_extrinsics = []           # 合并后的外参矩阵列表
        self.combined_intrinsics = []           # 合并后的内参矩阵列表  
        self.combined_depth_maps = []           # 合并后的深度图列表
        self.combined_depth_confs = []          # 合并后的深度置信度列表
        self.combined_world_points = []         # 合并后的世界坐标点云
        self.combined_world_points_confs = []   # 合并后的点云置信度
        self.all_camera_poses = []              # 所有相机位姿 

class VGGT_Long:
    """
    VGGT-Long: 长序列图像处理的主要类
    
    该类实现了一个完整的长序列图像3D重建和SLAM流水线，包括：
    1. 图像分块处理以避免内存溢出
    2. VGGT模型推理获得深度和位姿
    3. 循环检测和闭合优化
    4. Sim3变换进行块间对齐
    5. 点云生成和相机轨迹可视化
    
    Args:
        image_dir (str): 输入图像目录路径
        save_dir (str): 结果保存目录路径  
        config (dict): 配置参数字典
    """
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        # 模型配置参数
        self.chunk_size = self.config['Model']['chunk_size']        # 每个块的图像数量
        self.overlap = self.config['Model']['overlap']              # 块间重叠的图像数量
        self.conf_threshold = 1.5                                   # 置信度阈值
        self.seed = 42                                              # 随机种子
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备
        # 根据GPU架构选择数据类型（A100及以上使用bfloat16，否则使用float16）
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False                                       # 是否使用天空掩码
        self.useDBoW = self.config['Model']['useDBoW']             # 是否使用DBoW2循环检测

        # 路径配置
        self.img_dir = image_dir                                    # 输入图像目录
        self.img_list = None                                        # 图像文件列表
        self.output_dir = save_dir                                  # 输出目录

        # 临时文件目录
        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')  # 未对齐结果
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')      # 已对齐结果
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')            # 循环检测结果
        self.pcd_dir = os.path.join(save_dir, 'pcd')                                  # 点云文件目录
        
        # 创建所有必要的目录
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []                                  # 存储所有相机位姿

        self.delete_temp_files = self.config['Model']['delete_temp_files']  # 是否删除临时文件

        print('Loading model...')

        # 加载VGGT模型
        self.model = VGGT()
        # 从本地文件加载预训练权重
        _URL = self.config['Weights']['VGGT']
        state_dict = torch.load(_URL, map_location='cuda')
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()                                           # 设置为评估模式
        self.model = self.model.to(self.device)                    # 移动到指定设备

        self.skyseg_session = None                                  # 天空分割模型会话

        # 天空分割相关代码（暂时注释）
        # if self.sky_mask:
        #     print('Loading skyseg.onnx...')
        #     # Download skyseg.onnx if it doesn't exist
        #     if not os.path.exists("skyseg.onnx"):
        #         print("Downloading skyseg.onnx...")
        #         download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")
        #     self.skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        
        # 数据结构初始化
        self.chunk_indices = None                                   # 存储块的索引范围 [(begin_idx, end_idx), ...]
        self.loop_list = []                                         # 循环检测结果 [(frame1, frame2), ...]
        self.loop_optimizer = Sim3LoopOptimizer(self.config)       # Sim3循环优化器
        self.sim3_list = []                                         # Sim3变换列表 [(s, R, T), ...]
        self.loop_sim3_list = []                                    # 循环相关的Sim3变换
        self.loop_predict_list = []                                 # 循环预测结果列表
        self.loop_enable = self.config['Model']['loop_enable']     # 是否启用循环检测

        # 初始化循环检测器
        if self.loop_enable:
            if self.useDBoW:
                # 使用DBoW2方法
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                # 使用DNIO v2方法
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def get_loop_pairs(self):
        """
        获取循环闭合帧对
        
        该方法支持两种循环检测方法：
        1. DBoW2: 基于词袋模型的图像检索方法
        2. DNIO v2: 基于深度学习的循环检测方法
        
        处理流程：
        - 遍历所有输入图像
        - 对每一帧进行循环检测
        - 将检测到的循环对添加到loop_list中
        """

        if self.useDBoW: # 使用DBoW2方法
            # 遍历所有图像进行循环检测
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                # 加载并预处理图像
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # 灰度图转换为RGB图像
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                # 缩放图像以提高处理速度
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                
                # 将当前帧添加到DBoW2数据库中
                self.retrieval(frame, frame_id)
                
                # 检测循环闭合
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # 例如: cands = (812, 67)
                    # 确认循环闭合
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                # 保存当前帧的检索信息
                self.retrieval.save_up_to(frame_id)

        else: # 使用DNIO v2方法
            # 运行DNIO v2循环检测器
            self.loop_detector.run()
            # 获取检测到的循环对列表
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        """
        处理单个图像块
        
        Args:
            range_1 (tuple): 第一个图像范围 (start_idx, end_idx)
            chunk_idx (int, optional): 块索引，用于常规处理
            range_2 (tuple, optional): 第二个图像范围，用于循环检测
            is_loop (bool): 是否是循环检测处理
            
        Returns:
            dict or None: 如果是循环处理或有range_2，返回预测结果；否则返回None
            
        功能说明：
        - 加载指定范围的图像
        - 使用VGGT模型进行推理
        - 将位姿编码转换为外参和内参矩阵
        - 保存结果到磁盘以节省内存
        """
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        
        # 如果有第二个范围，合并图像路径（用于循环检测）
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # 加载和预处理图像
        images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(images)} images")
        
        # 验证图像格式: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 使用VGGT模型进行推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images)
        torch.cuda.empty_cache()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        # 将位姿编码转换为外参和内参矩阵
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        print("Processing model outputs...")
        # 将GPU张量转换为CPU numpy数组
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # 确定保存路径和文件名
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        
        save_path = os.path.join(save_dir, filename)
                    
        # 存储相机位姿信息（仅对常规块处理）
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))

        # 压缩深度图维度
        predictions['depth'] = np.squeeze(predictions['depth'])

        # 保存预测结果到磁盘
        np.save(save_path, predictions)
        
        # 根据处理类型返回结果
        return predictions if is_loop or range_2 is not None else None
    
    def process_long_sequence(self):
        """
        处理长序列图像的主要方法
        
        该方法实现了完整的长序列图像处理流水线：
        1. 将图像序列分割成重叠的块
        2. 处理循环检测（如果启用）
        3. 对每个块进行VGGT推理
        4. 计算块间的Sim3对齐变换
        5. 执行循环闭合优化（如果有循环）
        6. 应用变换对齐所有块
        7. 生成点云文件
        8. 保存相机轨迹
        
        异常处理：
        - 检查重叠量不能大于等于块大小
        - 单独处理只有一个块的简单情况
        """
        # 验证配置参数
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        
        # 计算块的划分
        if len(self.img_list) <= self.chunk_size:
            # 图像数量少于块大小，只需一个块
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            # 计算重叠块的划分
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))
        
        # 处理循环检测和Sim3估计
        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            # 处理循环列表，生成循环相关的块范围
            loop_results = process_loop_list(self.chunk_indices, 
                                             self.loop_list, 
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            
            # 对每个循环对进行处理
            # 返回格式示例: (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                # 处理循环相关的图像块
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        # 处理每个图像块
        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        # 释放模型内存
        del self.model # Save GPU Memory
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        # 计算相邻块之间的Sim3变换
        for chunk_idx in range(len(self.chunk_indices)-1):

            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            # 加载相邻两个块的数据
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            # 提取重叠区域的点云数据
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            # 计算置信度阈值
            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            # 使用加权点云对齐算法计算Sim3变换
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2, 
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))

        # 处理循环闭合的Sim3变换
        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                # 处理块A的对齐
                print('chunk_a align')
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]
            
                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                          conf_a, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                # 处理块B的对齐  
                print('chunk_a align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]
            
                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                          conf_b, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                # 计算A到B的Sim3变换
                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        # 执行循环闭合优化
        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                """提取位姿的XYZ坐标"""
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # 可视化优化结果
            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        # 应用对齐变换
        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(f'Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})')
            s, R, t = self.sim3_list[chunk_idx]
            
            # 加载块数据并应用变换
            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)
            
            # 保存对齐后的数据
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, chunk_data)
            
            # 处理第一个块（不需要变换）
            if chunk_idx == 0:
                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"), allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            
            # 加载对齐后的数据
            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first
            
            # 生成点云文件
            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,              # shape: (H, W, 3)
                colors=colors,              # shape: (H, W, 3)
                confs=confs,          # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        # 保存相机位姿
        self.save_camera_poses()
        
        print('Done.')

    def export_glb(self, glb_path=None, conf_thres=50.0, filter_by_frames="All", mask_black_bg=False, mask_white_bg=False, show_cam=True, mask_sky=False, prediction_mode="Predicted Pointmap"):
        """
        导出当前处理结果为 glb 3D 模型文件
        Args:
            glb_path (str): glb 文件保存路径，默认为 output_dir/vggt_long_result.glb
            其余参数同 predictions_to_glb
        """
        # 读取最后一个块的预测结果（或合并所有块）
        # 这里只做简单示例，实际可根据需要合并所有块的点云
        for idx in reversed(range(len(self.chunk_indices))):
            chunk_path = os.path.join(self.result_aligned_dir, f"chunk_{idx}.npy")
            if os.path.exists(chunk_path):
                break
        else:
            print("[GLB导出] 未找到任何对齐后的块数据")
            return
        predictions = np.load(chunk_path, allow_pickle=True).item()
        # 生成 glb scene
        scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=filter_by_frames,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=self.output_dir,
            prediction_mode=prediction_mode
        )
        # 保存 glb 文件
        if glb_path is None:
            glb_path = os.path.join(self.output_dir, "vggt_long_result.glb")
        scene.export(glb_path)
        print(f"[GLB导出] 已保存: {glb_path}")

    def export_merged_glb(self, glb_path=None, conf_thres=50.0, filter_by_frames="All", mask_black_bg=False, mask_white_bg=False, show_cam=True, mask_sky=False, prediction_mode="Predicted Pointmap"):
        """
        导出合并所有块数据的 glb 3D 模型文件
        
        Args:
            glb_path (str): glb 文件保存路径，默认为 output_dir/vggt_long_merged_result.glb
            其余参数同 predictions_to_glb
        """
        print("[GLB导出] 开始合并所有块数据...")
        
        # 检查是否有对齐后的数据
        aligned_files = []
        for idx in range(len(self.chunk_indices)):
            chunk_path = os.path.join(self.result_aligned_dir, f"chunk_{idx}.npy")
            if os.path.exists(chunk_path):
                aligned_files.append(chunk_path)
        
        if not aligned_files:
            print("[GLB导出] 未找到任何对齐后的块数据")
            return
        
        print(f"[GLB导出] 找到 {len(aligned_files)} 个对齐后的块文件")
        
        # 初始化合并数据结构
        combined_predictions = {
            'world_points': [],
            'world_points_conf': [],
            'images': [],
            'extrinsic': [],
            'intrinsic': [],
            'depth': []
        }
        
        # 合并所有块的数据
        for i, chunk_path in enumerate(aligned_files):
            print(f"[GLB导出] 正在处理块 {i+1}/{len(aligned_files)}: {os.path.basename(chunk_path)}")
            chunk_data = np.load(chunk_path, allow_pickle=True).item()
            
            # 检查数据键是否存在
            for key in combined_predictions.keys():
                if key in chunk_data:
                    if isinstance(chunk_data[key], np.ndarray):
                        combined_predictions[key].append(chunk_data[key])
                    else:
                        print(f"[警告] 块 {i} 中的 {key} 不是numpy数组，跳过")
                else:
                    print(f"[警告] 块 {i} 中缺少键: {key}")
        
        # 将列表转换为numpy数组
        for key in combined_predictions.keys():
            if combined_predictions[key]:
                try:
                    combined_predictions[key] = np.concatenate(combined_predictions[key], axis=0)
                    print(f"[GLB导出] 合并后 {key} 形状: {combined_predictions[key].shape}")
                except Exception as e:
                    print(f"[警告] 合并 {key} 时出错: {e}")
                    # 如果合并失败，使用第一个块的数据
                    if combined_predictions[key]:
                        combined_predictions[key] = combined_predictions[key][0]
            else:
                print(f"[警告] {key} 数据为空")
        
        # 生成 glb scene
        print("[GLB导出] 生成GLB场景...")
        scene = predictions_to_glb(
            combined_predictions,
            conf_thres=conf_thres,
            filter_by_frames=filter_by_frames,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=self.output_dir,
            prediction_mode=prediction_mode
        )
        
        # 保存 glb 文件
        if glb_path is None:
            glb_path = os.path.join(self.output_dir, "vggt_long_merged_result.glb")
        
        print(f"[GLB导出] 保存GLB文件到: {glb_path}")
        scene.export(glb_path)
        print(f"[GLB导出] 合并数据GLB文件已保存: {glb_path}")

    
    def run(self):
        """
        运行完整的VGGT-Long处理流水线
        
        主要执行步骤：
        1. 加载图像文件列表
        2. 执行循环检测（如果启用）
        3. 释放循环检测器内存
        4. 处理长序列图像
        
        异常处理：
        - 检查图像目录是否为空
        - 支持jpg和png格式的图像
        """
        print(f"Loading images from {self.img_dir}...")
        # 搜索图像文件（支持jpg和png格式）
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
                                glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)  # 调试用，可以打印图像列表
        
        # 检查是否找到图像文件
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        # 执行循环检测（如果启用）
        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                # 释放DBoW2检索器内存
                self.retrieval.close() # Save CPU Memory
                gc.collect()
            else:
                # 释放DNIO v2检测器内存
                del self.loop_detector # Save GPU Memory
        
        # 清理GPU缓存
        torch.cuda.empty_cache()

        # 处理长序列图像
        self.process_long_sequence()

    def save_camera_poses(self):
        """
        保存所有块的相机位姿到txt和ply文件
        
        输出文件：
        - txt文件: 每行包含一个4x4的C2W矩阵（展平为16个数字）
        - ply文件: 相机位姿可视化为点，每个块使用不同颜色
        
        处理流程：
        1. 初始化不同颜色用于区分块
        2. 处理第一个块的位姿（无需变换）
        3. 对其他块应用累积的Sim3变换
        4. 将所有位姿保存为txt格式
        5. 生成ply格式的可视化文件
        """
        # 定义不同块的颜色
        chunk_colors = [
            [255, 0, 0],    # 红色
            [0, 255, 0],    # 绿色
            [0, 0, 255],    # 蓝色
            [255, 255, 0],  # 黄色
            [255, 0, 255],  # 洋红色
            [0, 255, 255],  # 青色
            [128, 0, 0],    # 深红色
            [0, 128, 0],    # 深绿色
            [0, 0, 128],    # 深蓝色
            [128, 128, 0],  # 橄榄色
        ]
        print("Saving all camera poses to txt file...")
        
        # 初始化位姿数组
        all_poses = [None] * len(self.img_list)
        
        # 处理第一个块（参考块，无需变换）
        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i] 
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w

        # 处理其他块（应用累积的Sim3变换）
        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            # 注意：调用save_camera_poses时，所有sim3变换已对齐到第一个块
            s, R, t = self.sim3_list[chunk_idx-1]   
            
            # 构建Sim3变换矩阵
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i]
                c2w = np.linalg.inv(w2c)

                # 应用Sim3变换（注意左乘）
                transformed_c2w = S @ c2w  # Be aware of the left multiplication!

                all_poses[idx] = transformed_c2w

        # 保存位姿到txt文件
        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')
        
        print(f"Camera poses saved to {poses_path}")
        
        # 生成ply格式的可视化文件
        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # 写入PLY文件头
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            # 写入顶点数据（相机位置和颜色）
            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')
        
        print(f"Camera poses visualization saved to {ply_path}")

    
    def close(self):
        """
        清理临时文件并计算回收的磁盘空间
        
        该方法删除处理过程中生成的所有临时文件，包括三个目录：
        - 未对齐结果目录
        - 已对齐结果目录  
        - 循环检测结果目录
        
        性能说明：
        - 每个子地图通常占用约350 MiB的磁盘空间
        - 对于4000张图像的输入流，临时文件总计可消耗60-90 GiB存储空间
        - 处理完成后进行清理对于防止不必要的磁盘空间占用至关重要
        
        Returns:
            无返回值，但会打印回收的磁盘空间大小
        """
        # 如果配置为不删除临时文件，则直接返回
        if not self.delete_temp_files:
            return
        
        total_space = 0

        # 删除未对齐结果目录中的文件
        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        # 删除已对齐结果目录中的文件
        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        # 删除循环检测结果目录中的文件
        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        
        print('Deleting temp files done.')

        # 显示回收的磁盘空间（以GiB为单位）
        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


if __name__ == '__main__':
    """
    主程序入口
    
    功能：
    1. 解析命令行参数
    2. 加载配置文件
    3. 创建实验目录
    4. 运行VGGT-Long处理流水线
    5. 清理资源和合并点云
    
    命令行参数：
    - --image_dir: 必需，输入图像目录路径
    - --config: 可选，配置文件路径（默认为'./configs/base_config.yaml'）
    """

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='Image path')
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)

    # 设置输出目录
    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    # 创建基于时间戳的实验目录
    save_dir = os.path.join(
            exp_dir, image_dir.replace("/", "_"), current_datetime
        )
    
    # 备用目录命名方案（注释掉）
    # save_dir = os.path.join(
    #     exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
    # )

    # 创建实验目录
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    vggt_long = VGGT_Long(image_dir, save_dir, config)
    vggt_long.run()
    vggt_long.close()

    # 清理内存
    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    # 合并所有点云文件
    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('VGGT Long done.')
    
    # 正常退出程序
    sys.exit()