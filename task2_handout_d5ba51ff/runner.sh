docker build --tag task2 . && \
  docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task2
