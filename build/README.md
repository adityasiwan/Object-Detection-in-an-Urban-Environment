# Instructions

## Requirements

* NVIDIA GPU with the latest driver installed
* docker / nvidia-docker

## Build
Build the image with:
```
docker build -t project-dev -f Dockerfile.gpu .
```

Create a container with:
```
docker run -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ -ti project-dev bash
```

```
docker run -p 8888:8888 --gpus all -v /home/aditya/Downloads/starter:/app/project/ -ti adityaef/udacity:new bash
```
and any other flag you find useful to your system (eg, `--shm-size`).

## Set up

Once in container, you will need to install gsutil, which you can easily do by running:
```
curl https://sdk.cloud.google.com | bash
```

Once gsutil is installed and added to your path, you can auth using:
```
gcloud auth login
```

## Debug
* Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) if you run into any issue with the installation of the
tf object detection api




```
docker start be83c4afc784
docker exec -it 4ae5a3f6301c bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

docker run -it -p 8888:8888 --gpus all -v /mnt/c/Users/Aditya/udacity/starter:/app/project/ -ti adityaef/udacity:new bash