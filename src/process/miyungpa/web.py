# dataset_viewer.py
from flask import Flask, render_template_string
from pathlib import Path
import os

app = Flask(__name__)

BASE_PATH = Path("/home/temp_id/shared_data/capture/miyungpa")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Viewer</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .container { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }
        .item { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .item img { width: 100%; height: auto; border-radius: 4px; cursor: pointer; }
        .item img:hover { opacity: 0.9; }
        .item h3 { margin: 10px 0 5px 0; font-size: 18px; }
        .item p { margin: 5px 0; color: #666; font-size: 14px; }
        .filter { margin: 20px 0; }
        .filter input { padding: 8px; width: 300px; font-size: 14px; }
        
        /* 이미지 클릭 시 전체화면 */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); }
        .modal img { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 90%; max-height: 90%; }
        .close { position: absolute; top: 20px; right: 40px; color: white; font-size: 40px; cursor: pointer; }
    </style>
    <script>
        function filterItems() {
            const input = document.getElementById('search').value.toLowerCase();
            const items = document.getElementsByClassName('item');
            for (let item of items) {
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(input) ? '' : 'none';
            }
        }
        
        function openModal(src) {
            document.getElementById('modal').style.display = 'block';
            document.getElementById('modalImg').src = src;
        }
        
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
    </script>
</head>
<body>
    <h1>Dataset Viewer - {{ total }} items</h1>
    <div class="filter">
        <input type="text" id="search" onkeyup="filterItems()" placeholder="Search object or date...">
    </div>
    <div class="container">
        {% for item in items %}
        <div class="item">
            <img src="/image/{{ item.obj_name }}/{{ item.date }}" alt="thumbnail" onclick="openModal(this.src)">
            <h3>{{ item.obj_name }}</h3>
            <p>Date: {{ item.date }}</p>
        </div>
        {% endfor %}
    </div>
    
    <div id="modal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img id="modalImg">
    </div>
</body>
</html>
"""

def scan_dataset():
    items = []
    if not BASE_PATH.exists():
        return items
    
    for obj_dir in BASE_PATH.iterdir():
        if not obj_dir.is_dir():
            continue
        obj_name = obj_dir.name
        
        for date_dir in obj_dir.iterdir():
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            
            thumb_path = date_dir / "thumbnail.jpg"
            if thumb_path.exists():
                items.append({
                    'obj_name': obj_name,
                    'date': date,
                    'path': str(thumb_path)
                })
    
    return sorted(items, key=lambda x: (x['obj_name'], x['date']))

@app.route('/')
def index():
    items = scan_dataset()
    return render_template_string(HTML_TEMPLATE, items=items, total=len(items))

@app.route('/image/<obj_name>/<date>')
def serve_image(obj_name, date):
    from flask import send_file
    img_path = BASE_PATH / obj_name / date / "thumbnail.jpg"
    if img_path.exists():
        return send_file(img_path, mimetype='image/jpeg')
    return "Image not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)