# Docker Instructions for HeSPaS

The following instruction setup a docker image for using the profiling-based
`xla_estimator`.

## Build the Image

```bash
docker build -t hespas .
```

## Run a Container

```bash
docker run --gpus all -it --rm hespas
```

| Flag | Description |
|------|-------------|
| `--gpus all` | Enable GPU access (remove if not needed) |
| `-it` | Interactive terminal |
| `--rm` | Remove container on exit |

### Mount Local Directory

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace hespas
```

## Useful Commands

### Attach to a running container
```
docker exec -it <container_id> /bin/bash
```
