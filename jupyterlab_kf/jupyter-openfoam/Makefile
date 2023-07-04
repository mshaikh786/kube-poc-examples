NB_USER?=training01
NB_UID?=882003
TAG?=${NB_USER}
DOCKERHUB_USER?=mshaikh
docker-build-base:
	make docker-build -C ../base TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-base
	docker build -t kf-openfoam:${TAG} --build-arg BASE_IMG=base:${TAG}  -f Dockerfile . 

docker-push-cpu:
	docker run -d --name kf-of kf-openfoam:${TAG}
	docker commit -m "Kubeflow image for OpenFOAM" kf-of ${DOCKERHUB_USER}/kf-openfoam:${TAG}
	docker push ${DOCKERHUB_USER}/kf-openfoam:${TAG}
	docker stop kf-of
	docker rm kf-of

