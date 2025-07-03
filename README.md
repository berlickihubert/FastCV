# Structure from motion images/video to 3d model converter


## Prerequisites
- Docker installed on your system

## Build and Run

1. Build the Docker image:
   ```powershell
   docker build -t cuda-demo .
   ```
2. Run the container:
   ```powershell
   docker run --gpus all --rm -v ${PWD}/resources/test_images:/app/resources/test_images cuda-demo
   ```