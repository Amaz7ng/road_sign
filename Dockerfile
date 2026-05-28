# Указываем легкую базовую версию Python
FROM python:3.10-slim

# Создаем рабочую папку внутри сервера
WORKDIR /code

# Копируем файл зависимостей и устанавливаем их
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Копируем все остальные файлы проекта (код, веса .pth)
COPY . .

# Команда для запуска FastAPI на Hugging Face (порт должен быть 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]