NB_USER?=training01
NB_UID?=882003
TAG?=${NB_USER}
DOCKERHUB_USER?=mshaikh
docker-build-base:
	make docker-build -C ../base_nocuda TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-base
	DOCKER_BUILDKIT=1 docker build -t kf-openfoam:${TAG} --build-arg BASE_IMG=base_nocuda:${TAG}  -f Dockerfile .

docker-push:
	docker run -d --name kf-of kf-openfoam:${TAG}
	docker commit -m "Kubeflow image for OpenFOAM" kf-of ${DOCKERHUB_USER}/kubeflow-demo:jupyter_OpenFOAM_${NB_USER}
	docker push ${DOCKERHUB_USER}/kubeflow-demo:jupyter_OpenFOAM_${NB_USER}
	docker stop kf-of
	docker rm kf-of

