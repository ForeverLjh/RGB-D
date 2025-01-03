import os
import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# 用户定义文件夹名称
folder_name = "20250104"

# 创建一级文件夹（如果不存在）
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 在一级文件夹下创建三个子文件夹
color_folder = os.path.join(folder_name, 'color_image')
depth_folder = os.path.join(folder_name, 'depth_image')
pointcloud_folder = os.path.join(folder_name, 'point_cloud')

# 创建子文件夹（如果不存在）
for folder in [color_folder, depth_folder, pointcloud_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()

# 配置相机参数，提升分辨率
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 启动相机数据流
pipeline.start(config)

# 添加对齐对象，将深度对齐到颜色流
align_to = rs.stream.color
align = rs.align(align_to)

# 添加深度滤波器
decimation_filter = rs.decimation_filter()  # 降采样滤波
spatial_filter = rs.spatial_filter()  # 空间滤波
temporal_filter = rs.temporal_filter()  # 时间滤波
hole_filling_filter = rs.hole_filling_filter()  # 填充空洞

try:
    while True:
        # 等待帧并获取数据
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到颜色帧
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 对深度帧应用滤波器
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 显示 RGB 和深度图像
        cv2.imshow("Aligned RGB Image", color_image)
        cv2.imshow("Aligned Filtered Depth Image", depth_image)

        # 按下 's' 键保存图像和点云数据
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 保存对齐后的 RGB 图像
            color_image_path = os.path.join(color_folder, "aligned_color_image.png")
            cv2.imwrite(color_image_path, color_image)

            # 保存对齐后的深度图像
            depth_image_path = os.path.join(depth_folder, "aligned_depth_image.png")
            cv2.imwrite(depth_image_path, depth_image)

            # 生成点云文件
            profile = depth_frame.profile.as_video_stream_profile()
            intrinsics = profile.get_intrinsics()
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy
            print(fx, fy, cx, cy, sep=" ")

            # 计算点云
            height, width = depth_image.shape
            points = []
            colors = []
            for y in range(height):
                for x in range(width):
                    z = depth_image[y, x] * 0.001  # 深度值转换为米
                    if z > 0:
                        # 根据内参计算点云坐标
                        X = (x - cx) * z / fx
                        Y = (y - cy) * z / fy
                        points.append([X, Y, z])

                        # 获取对应的 RGB 值
                        colors.append(color_image[y, x] / 255.0)  # 归一化颜色

            # 将点云转换为 Open3D 格式
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

            # 可选：对点云降采样
            point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

            # 保存点云文件
            point_cloud_path = os.path.join(pointcloud_folder, "aligned_point_cloud.pcd")
            o3d.io.write_point_cloud(point_cloud_path, point_cloud)
            print("对齐的 RGB 图像、深度图像和点云文件已保存。")

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止数据流
    pipeline.stop()
    cv2.destroyAllWindows()
