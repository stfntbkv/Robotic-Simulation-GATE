from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='.')

BASE_PATHS = {
    'primitive': '/remote-home1/sdzhang/datasets/OpenRT/vlabench_task/primitive',
    'composite': '/remote-home1/sdzhang/datasets/OpenRT/vlabench_task/composite',
}

def count_hdf5_files(root_dir):
    result = {}
    for subdir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_path):
            count = sum(f.endswith('.hdf5') for f in os.listdir(full_path))
            result[subdir] = count
    return result

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
    app.run(host='0.0.0.0', port=8500)