### Run with Docker

```bash
docker build -t star-classifier .
docker run --gpus all \
  -v /path/to/your/data:/app/data \
  star-classifier python gpu.py --data_root /app/data
