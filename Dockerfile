# 基于 Python 的官方镜像作为基础镜像
FROM python:3.11.3

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . .

# 设置时区
ENV TZ=Asia/Shanghai

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 运行 Django 项目
CMD python manage.py runserver 0.0.0.0:8000
