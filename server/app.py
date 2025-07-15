from flask import Flask, request, jsonify, send_from_directory
import os
import h5py
import numpy as np
import base64
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python"])
    import cv2

app = Flask(__name__, static_folder='.')
BASE_PATHS = {
    'primitive': '/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vlabench/pretrain/primitive',
    'composite': '/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vlabench/pretrain/composite',
}

def list_tasks(task_type):
    root = BASE_PATHS[task_type]
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def list_hdf5_files(task_type, task):
    task_dir = os.path.join(BASE_PATHS[task_type], task)
    return [f for f in os.listdir(task_dir) if f.endswith('.hdf5')]

def load_instruction_and_video(filepath, n_frame=10):
    # default返回第一个timestamp里的内容
    try:
        with h5py.File(filepath, 'r') as f:
            data = f["data"]
            timestamps = list(data.keys())
            if not timestamps:
                return {"error": "No timestamps in file"}
            ts = timestamps[0]
            
            # 读取instruction和rgb图片
            instruction_raw = data[ts]["instruction"]
            
            # 处理instruction数据，确保可以JSON序列化
            if isinstance(instruction_raw, h5py.Dataset):
                instruction_array = np.array(instruction_raw)
                # 如果是字节串数组，需要解码
                if instruction_array.dtype.kind in ['S', 'U']:  # 字符串类型
                    if instruction_array.ndim == 0:  # 标量
                        instruction = str(instruction_array.item())
                        if isinstance(instruction_array.item(), bytes):
                            instruction = instruction_array.item().decode('utf-8')
                    else:  # 数组
                        instruction = []
                        for item in instruction_array.flat:
                            if isinstance(item, bytes):
                                instruction.append(item.decode('utf-8'))
                            else:
                                instruction.append(str(item))
                else:
                    # 其他类型转为列表
                    instruction = instruction_array.tolist()
            else:
                # 直接转换
                instruction = str(instruction_raw)
            
            rgbs = np.array(data[ts]["observation"]["rgb"])  # [T, N, C, H, W], uint8
            print(f"RGB shape: {rgbs.shape}")  # 调试信息
            
            n_frame = min(n_frame, rgbs.shape[0])
            n_views = rgbs.shape[1]  # 视角数量
            
            # 重新组织数据结构：按视角分组
            views_data = []
            
            for view_idx in range(n_views):
                view_frames = []
                for frame_idx in range(n_frame):
                    # rgbs[frame_idx, view_idx] 是 [C, H, W] 格式
                    img = rgbs[frame_idx, view_idx]  # [C, H, W]
                    
                    # 转换为 [H, W, C] 格式
                    if img.shape[0] == 3:  # 如果第一维是通道数
                        img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                    
                    # 转为JPEG再base64
                    ok, encimg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if ok:
                        b64 = base64.b64encode(encimg).decode('utf-8')
                        view_frames.append(b64)
                
                views_data.append({
                    'view_id': view_idx,
                    'frames': view_frames
                })
            
            return {
                "instruction": instruction,
                "total_frames": int(rgbs.shape[0]),
                "n_views": int(n_views),
                "views_data": views_data
            }
    except Exception as e:
        return {"error": str(e)}

def count_hdf5_files(root_dir):
    result = {}
    for subdir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_path):
            count = sum(f.endswith('.hdf5') for f in os.listdir(full_path))
            result[subdir] = count
    return result

@app.route('/tasks')
def get_tasks():
    task_type = request.args.get('type')
    if task_type not in BASE_PATHS:
        return jsonify([])  # invalid
    return jsonify(list_tasks(task_type))

@app.route('/files')
def get_files():
    task_type = request.args.get('type')
    task = request.args.get('task')
    if task_type not in BASE_PATHS or not task:
        return jsonify([])
    return jsonify(list_hdf5_files(task_type, task))

@app.route('/hdf5_content')
def get_hdf5_content():
    task_type = request.args.get('type')
    task = request.args.get('task')
    filename = request.args.get('file')
    n_frame = int(request.args.get('n_frame', 10))
    if task_type not in BASE_PATHS or not task or not filename:
        return jsonify({'error': 'bad request'})
    filepath = os.path.join(BASE_PATHS[task_type], task, filename)
    result = load_instruction_and_video(filepath, n_frame)
    return jsonify(result)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/stats')
def get_stats():
    task_type = request.args.get('type')
    if task_type not in BASE_PATHS:
        return jsonify({'error': 'invalid type'}), 400
    stats = count_hdf5_files(BASE_PATHS[task_type])
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8500, debug=True)