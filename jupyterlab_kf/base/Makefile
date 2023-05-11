NB_USER?=training01
NB_UID?=882003
TAG?=${NB_USER}
DOCKERHUB_USER?=mshaikh

docker-build:
	docker build -t base:${TAG} --build-arg NB_USER=${NB_USER} --build-arg NB_UID=${NB_UID} -f Dockerfile .

docker-push:
	docker push ${DOCKERHUB_USER}/base:${TAG}
