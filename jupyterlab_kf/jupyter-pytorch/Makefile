NB_USER?=training01
NB_UID?=882003
TAG?=gpu
DOCKERHUB_USER?=mshaikh


docker-build-jupyter:
	make docker-build -C ../jupyter TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-jupyter
	docker build -t jupyter-pytorch:${TAG} --build-arg BASE_IMG=jupyter-kf:${TAG}  -f Dockerfile.${TAG} . 

#
#	docker build -t jupyter-pytorch-cuda:${TAG} --build-arg BASE_IMG=jupyter:${TAG} -f cuda.Dockerfile . 

#docker-push-cpu:
#	docker push jupyter-pytorch:${TAG}

#docker-push-cuda:
#	docker push jupyter-pytorch-cuda:${TAG}
docker-push:
	docker tag jupyter-pytorch:${TAG} ${DOCKERHUB_USER}/kubeflow-demo:jupyter_pytorch_${TAG}
	docker push ${DOCKERHUB_USER}/kubeflow-demo:jupyter_pytorch_${TAG}
