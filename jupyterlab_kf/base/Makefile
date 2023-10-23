NB_USER?=training01
NB_UID?=882003
TAG?=gpu
DOCKERHUB_USER?=mshaikh

docker-build:
	DOCKER_BUILDKIT=1 docker build -t base:${TAG} --build-arg NB_USER=${NB_USER} --build-arg NB_UID=${NB_UID} -f Dockerfile.${TAG} .

docker-push:
	docker push ${DOCKERHUB_USER}/base:${TAG}
