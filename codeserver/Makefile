NB_USER?=shaima0d
NB_UID?=174988
TAG?=${NB_USER}
DOCKERHUB_USER?=mshaikh

docker-build-base:
	make docker-build -C ../base TAG=${TAG} NB_USER=${NB_USER} NB_UID=${NB_UID}

docker-build: docker-build-base
	docker build -t codeserver:${TAG} --build-arg BASE_IMG=base:${TAG} -f Dockerfile . 

docker-push:
	docker run -d --name codeserver-test codeserver:${TAG}
	docker commit -m 'codeserver with custom UID and GID' codeserver-test ${DOCKERHUB_USER}/codeserver:${TAG}
	docker push ${DOCKERHUB_USER}/codeserver:${TAG}
	docker stop codeserver-test
	docker rm codeserver-test
