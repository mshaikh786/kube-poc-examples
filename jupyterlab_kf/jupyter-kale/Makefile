NB_USER?=training01
NB_UID?=882003
TAG?=gpu
DOCKERHUB_USER?=mshaikh


docker-build-base:
	make docker-build -C ../base TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-base
	docker build -t jupyter_kale:${TAG} --build-arg BASE_IMG=base:${TAG} -f Dockerfile .

docker-push:
	docker tag jupyter_kale:${TAG} ${DOCKERHUB_USER}/kubeflow-demo:jupyter_kale_${TAG}
	docker push ${DOCKERHUB_USER}/kubeflow-demo:jupyter_kale_${TAG}
