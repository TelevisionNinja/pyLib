# pyLib
A collection of some python 3 functions

Subjects that this library covers:
- arrays
- math

Arrays:
- utilities to manipulate arrays

Math:
- general math functions
- integration methods
- root finding methods
- vector operations

# Run Tests
## Native
```bash
python -m unittest
```

## Docker or Podman
### Docker
```bash
docker build -t pylib_image .
docker run --replace --name pylib_container pylib_image
```

### Podman
```bash
podman build -t pylib_image .
podman run --replace --name pylib_container pylib_image
```
