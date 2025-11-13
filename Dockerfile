# image base
FROM tensorflow/tensorflow:2.15.0-gpu

#  install and config uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

# install deps
RUN uv pip install --system -r requirements.txt

COPY . .

CMD ["python", "model_train_dp.py"]