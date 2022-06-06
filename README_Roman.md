# Docker

## Run
docker run -it -u $(id -u):$(id -g) --gpus all -v /home/roman/projects/udacity-self-driving/projects/object_detection:/app/project/ --network=host project-dev

## Run in background
docker run --rm -dit -u $(id -u):$(id -g) --gpus all -v /home/roman/projects/udacity-self-driving/projects/object_detection:/app/project/ --network=host --name udacity-cv project-dev bash

docker run --rm -dit --gpus all -v /home/roman/projects/udacity-self-driving/projects/object_detection:/app/project/ --name udacity-cv project-dev bash

## Attach
docker attach
Leave container with Ctrl-p Ctrl-q


# vscode

rm -rf ~/.vscode ~/.cache/Code

