# ���� Python �Ĺٷ�������Ϊ��������
FROM python:3.11.3

# ���ù���Ŀ¼
WORKDIR /app

# ������Ŀ�ļ���������
COPY . .

# ����ʱ��
ENV TZ=Asia/Shanghai

# ���� Django ��Ŀ
CMD python manage.py runserver 0.0.0.0:8000