from PIL import Image
import os
print(os.getcwd())
#PIL.Image.open()、torch.load() 等第三方库函数，默认基准是「当前工作目录（CWD）」
# 步骤1：获取脚本所在目录的绝对路径
script_dir= os.path.dirname(os.path.abspath(__file__))# __file__ 是当前脚本的路径
# 步骤2：拼接相对路径（脚本目录 + 目标文件名）
file_path = os.path.join(script_dir, "pic/street.jpg")
image = Image.open(file_path)
iw, ih  = image.size#原始图像的宽和高

target_size = (iw*2, ih*3)
w, h    = target_size#目标图像的宽和高

scale   = min(w/iw, h/ih)#计算缩放比例，选择较小的比例以保持图像的宽高比
nw      = int(iw*scale)#新的宽度
nh      = int(ih*scale)#新的高度
image   = image.resize((nw,nh), Image.BICUBIC)
new_image = Image.new('RGB', target_size, (128,128,128))
new_image.paste(image, ((w-nw)//2, (h-nh)//2))

save_dir = os.path.join(script_dir, "pic")  # 保存目录路径
if not os.path.exists(save_dir):  # 若目录不存在，自动创建
    os.makedirs(save_dir)

# 步骤2：定义保存文件名和完整路径（避免覆盖原始文件）
save_filename = "street_resized.jpg"  # 自定义保存文件名
save_path = os.path.join(save_dir, save_filename)  # 完整保存路径

# 步骤3：保存图像（支持 jpg、png 等格式，根据后缀自动识别）
new_image.save(save_path)
print(f"图像已保存到：{save_path}")  # 打印保存路径，方便查看

# 指定输出文件名
output_file = 'requirementsnew.txt'

import pkg_resources

# 获取当前环境中已安装的所有模块
installed_packages = pkg_resources.working_set

# 将模块和版本信息存储在一个列表中
modules = []
for package in installed_packages:
    modules.append(f"{package.project_name}=={package.version}")

# 打印模块信息（可选）
for module in modules:
    print(module)



# 将模块信息写入到文件中
with open(output_file, 'w') as file:
    for module in modules:
        file.write(module + '\n')  # 每个模块信息写入一行

print(f"模块版本信息已保存到 {output_file}")