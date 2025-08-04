import cv2
import os
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
import hashlib
from PIL import Image


# # 基本使用
# python video_keyframe_extractor.py --video input.mp4 --output ./keyframes

# # 自定义参数
# python video_keyframe_extractor.py --video input.mp4 --output ./keyframes --threshold 0.80 --skip 2

class VideoKeyFrameExtractor:
    def __init__(self, video_path, output_dir, similarity_threshold=0.85, skip_frames=5):
        """
        视频关键帧提取器
        
        Args:
            video_path: MP4视频文件路径
            output_dir: 输出图片保存目录
            similarity_threshold: 相似度阈值（0-1），超过此值则跳过该帧
            skip_frames: 跳帧数，每隔几帧检测一次
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        self.skip_frames = skip_frames
        self.previous_frame = None
        self.frame_count = 0
        self.saved_count = 0
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
    
    def calculate_frame_similarity(self, frame1, frame2):
        """
        计算两帧之间的相似度
        使用结构相似性指数(SSIM)
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 调整尺寸以提高计算速度
        height, width = gray1.shape
        if height > 480 or width > 640:
            new_height = min(480, height)
            new_width = min(640, width)
            gray1 = cv2.resize(gray1, (new_width, new_height))
            gray2 = cv2.resize(gray2, (new_width, new_height))
        
        # 计算SSIM
        similarity, _ = ssim(gray1, gray2, full=True)
        return similarity
    
    def calculate_histogram_similarity(self, frame1, frame2):
        """
        计算直方图相似度（备用方法）
        """
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        # 使用相关性比较
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation
    
    def save_frame(self, frame, frame_number):
        """
        保存帧为图片文件
        """
        filename = f"keyframe_{self.saved_count:06d}_{frame_number:08d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # 转换BGR到RGB（OpenCV使用BGR，但PIL使用RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.save(filepath, quality=95)
        
        self.saved_count += 1
        print(f"保存关键帧: {filename}")
    
    def extract_keyframes(self):
        """
        提取关键帧
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {self.video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"视频信息:")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 帧率: {fps:.2f} FPS")
        print(f"  - 时长: {duration:.2f} 秒")
        print(f"  - 相似度阈值: {self.similarity_threshold}")
        print(f"  - 跳帧间隔: {self.skip_frames}")
        print("开始提取关键帧...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # 跳帧处理
                if self.frame_count % (self.skip_frames + 1) != 0:
                    continue
                
                # 第一帧直接保存
                if self.previous_frame is None:
                    self.save_frame(frame, self.frame_count)
                    self.previous_frame = frame.copy()
                    continue
                
                # 计算与前一帧的相似度
                similarity = self.calculate_frame_similarity(frame, self.previous_frame)
                
                # 如果相似度低于阈值，则保存为关键帧
                if similarity < self.similarity_threshold:
                    self.save_frame(frame, self.frame_count)
                    self.previous_frame = frame.copy()
                    
                # 显示进度
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        finally:
            cap.release()
        
        print(f"\n提取完成!")
        print(f"  - 处理帧数: {self.frame_count}")
        print(f"  - 关键帧数: {self.saved_count}")
        print(f"  - 压缩比例: {(1 - self.saved_count/total_frames)*100:.1f}%")
        print(f"  - 输出目录: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='MP4视频关键帧提取器')
    parser.add_argument('--video', type=str, required=True, help='输入MP4视频文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出图片保存目录')
    parser.add_argument('--threshold', type=float, default=0.85, help='相似度阈值 (0-1, 默认0.85)')
    parser.add_argument('--skip', type=int, default=5, help='跳帧间隔 (默认5)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在 {args.video}")
        return
    
    if not args.video.lower().endswith('.mp4'):
        print("警告: 输入文件不是.mp4格式，可能无法正确处理")
    
    # 创建提取器并执行
    extractor = VideoKeyFrameExtractor(
        video_path=args.video,
        output_dir=args.output,
        similarity_threshold=args.threshold,
        skip_frames=args.skip
    )
    
    extractor.extract_keyframes()

if __name__ == '__main__':
    main()