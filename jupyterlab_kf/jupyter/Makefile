NB_USER?=training01
NB_UID?=882003
TAG?=gpu
DOCKERHUB_USER?=mshaikh
BASE?=base

docker-build-base:
	make docker-build -C ../${BASE} TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-base
	docker build -t jupyter-kf:${TAG} --build-arg BASE_IMG=${BASE}:${TAG} -f Dockerfile .

#docker-push:
 #   docker tag jupyter-kf:${TAG} ${DOCKERHUB_USER}/kubeflow-demo:jupyter_${TAG}
#	docker push ${DOCKERHUB_USER}/kubeflow-demo:jupyter_${TAG}
