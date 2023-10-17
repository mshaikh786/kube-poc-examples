NB_USER?=training01
NB_UID?=882003
TAG?=gpu
DOCKERHUB_USER?=mshaikh
BASE?=base
docker-build:
	DOCKER_BUILDKIT=1 docker build -t ${BASE}:${TAG} --build-arg NB_USER=${NB_USER} --build-arg NB_UID=${NB_UID} -f Dockerfile .

docker-push:
	docker push ${DOCKERHUB_USER}/base_nocuda:${TAG}
