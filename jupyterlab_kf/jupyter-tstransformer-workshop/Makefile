NB_USER?=jovyan
NB_UID?=1000
TAG?=cpu
DOCKERHUB_USER?=mshaikh


docker-build-base:
	make docker-build -C ../base TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-base
	docker build -t jupyter-tstf:${TAG} --build-arg BASE_IMG=base:${TAG} -f Dockerfile .

docker-push:
	docker tag jupyter-tstf:${TAG} ${DOCKERHUB_USER}/kubeflow-demo:jupyter_tstf${TAG}
	docker push ${DOCKERHUB_USER}/kubeflow-demo:jupyter_tstf${TAG}
