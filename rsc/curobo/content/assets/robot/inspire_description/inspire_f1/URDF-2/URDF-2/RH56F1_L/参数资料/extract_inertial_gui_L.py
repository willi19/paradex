import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from docx import Document
import re

def extract_data_from_docx(file_name):
    doc = Document(file_name)
    raw_text = []

    for para in doc.paragraphs:
        raw_text.append(para.text)
    
    raw_string = "\n".join(raw_text)
    
    # 按照“数字+冒号”分割段落
    sections = re.split(r'\n(?=\d+:)', raw_string.strip())

    outputs = []
    for section in sections:
        output = process_section(section)
        if output:
            outputs.append(output)
    return outputs

def process_section(section):
    try:
        # 提取描述内容（如"整手"）
        description = extract_description(section)
        # 提取参数
        mass = extract_value(section, "质量") / 1000
        cog_x = extract_value(section, "X") / 1000
        cog_y = extract_value(section, "Y") / 1000
        cog_z = extract_value(section, "Z") / 1000
        
        ixx = extract_value(section, "Lxx") / 1e9
        ixy = extract_value(section, "Lxy") / 1e9
        ixz = -extract_value(section, "Lxz") / 1e9
        iyy = extract_value(section, "Lyy") / 1e9
        iyz = extract_value(section, "Lyz") / 1e9
        izz = extract_value(section, "Lzz") / 1e9

        output_string = f"""{description}.
#########################
    <inertial>
      <origin
        xyz="{cog_x} {cog_y} {cog_z}"
        rpy="0 0 0" />
      <mass
        value="{mass}" />
      <inertia
        ixx="{ixx}"
        ixy="{ixy}"
        ixz="{ixz}"
        iyy="{iyy}"
        iyz="{iyz}"
        izz="{izz}" />
    </inertial>
"""
        return output_string
    except ValueError as e:
        print(f"处理段落时出错: {e}")
        return None

def extract_value(section, key):
    pattern = re.compile(rf"{key}\s*=\s*([\d\.\-e]+)")
    match = pattern.search(section)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"未找到键: {key}")

def extract_description(section):
    # 匹配“数字: 内容.”格式
    match = re.search(r'^\d+:\s*(.+?)\.', section, re.MULTILINE)
    if match:
        return match.group(1).strip()
    else:
        return "未识别"

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Word Files", "*.docx")])
    if file_path:
        try:
            outputs = extract_data_from_docx(file_path)
            result_text.delete(1.0, tk.END)
            for i, output in enumerate(outputs, start=1):
                result_text.insert(tk.END, f"{i}: {output.strip()}\n#########################\n\n")
        except Exception as e:
            messagebox.showerror("错误", f"处理文件时出错: {e}")

# 查找功能
def search_text():
    keyword = simpledialog.askstring("搜索", "输入要查找的关键词：")
    if not keyword:
        return
    content = result_text.get(1.0, tk.END)
    start_idx = content.find(keyword)
    if start_idx == -1:
        messagebox.showinfo("未找到", f"未找到关键词：{keyword}")
        return
    # 转换位置到Text控件索引
    line_start = content.count("\n", 0, start_idx) + 1
    col_start = start_idx - content.rfind("\n", 0, start_idx) - 1
    line_end = line_start
    col_end = col_start + len(keyword)
    # 设置高亮
    result_text.tag_remove("search", "1.0", tk.END)
    result_text.tag_add("search", f"{line_start}.{col_start}", f"{line_end}.{col_end}")
    result_text.tag_config("search", background="yellow")
    result_text.see(f"{line_start}.{col_start}")

# 全选功能
def select_all():
    result_text.tag_add("sel", "1.0", tk.END)
    result_text.tag_config("sel", background="lightblue")
    result_text.focus_set()

# 创建GUI
root = tk.Tk()
root.title("urdf格式转换器")

# 按钮横向排列的Frame容器
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# 上传按钮
upload_button = tk.Button(button_frame, text="上传 Word 文件", command=upload_file)
upload_button.pack(side=tk.LEFT, padx=5)

# 查找按钮
search_button = tk.Button(button_frame, text="查找关键词", command=search_text)
search_button.pack(side=tk.LEFT, padx=5)

# 全选按钮
select_all_button = tk.Button(button_frame, text="全选", command=select_all)
select_all_button.pack(side=tk.LEFT, padx=5)

# 结果显示区
result_text = tk.Text(root, width=80, height=30)
result_text.pack(pady=10)

# 绑定快捷键（Ctrl+F调用搜索）
root.bind('<Control-f>', lambda event: search_text())
# 绑定快捷键（Ctrl+A调用全选）
root.bind('<Control-a>', lambda event: select_all())

root.mainloop()

