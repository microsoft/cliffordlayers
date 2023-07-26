# Installation Guide

```bash
pip install cliffordlayers
```

## For development

```bash title="clone the repo"
git clone https://github.com/microsoft/cliffordlayers
```

=== "`conda`"
    ```bash title="create and activate env"
    cd cliffordlayers
    conda env create --file docker/environment.yml
    conda activate cliffordlayers
    ```

    ```bash title="make an editable install"
    pip install -e .
    ```

=== "`docker`"
    ```bash title="build docker container"
    cd cliffordlayers/docker
    docker build -t cliffordlayers .
    cd ..
    ```

    ```bash title="run docker container"
    docker run --gpus all -it --rm --user $(id -u):$(id -g) \
        -v $(pwd):/code  --workdir /code -e PYTHONPATH=/code \
        cliffordlayers:latest
    ```
