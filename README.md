# mapper_speedrun

Speedrun version of the local mapper, made in Python.

## Requirements
- ROS Humble
- ONNX Runtime (onnxruntime)
- OpenCV
- NumPy

OR 

- Docker

## Building
### Docker (recommended for development)
- Build the container with `docker build -t mapper .`.
- Run the conntainer with `docker run --rm -it --gpus all --network=host --ipc=host mapper`.

### Bare metal
TODO
