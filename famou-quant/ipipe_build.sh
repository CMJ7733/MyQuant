#!/usr/bin/env bash
set -ex

CI_STAGE=${CI_STAGE}
WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Build ci, mode ${CI_STAGE}, work_root ${WORK_ROOT}"
cd "${WORK_ROOT}"

IMAGE_TAG=""
VERSION=$(echo "${AGILE_COMPILE_BRANCH}" | cut -d'/' -f2)
if [ "${VERSION}" == "online" ]; then
    IMAGE_TAG="${VERSION}.${AGILE_PIPELINE_BUILD_NUMBER}"
fi

prepare_output() {
    rm -rf output
    mkdir -p output/fm-agent/
    cp -rf famou/ output/fm-agent/
    cp -rf api_server/ output/fm-agent/
    cp -rf entrypoint_fm.sh output/
    cp -rf Dockerfile_ipipe output/
    cp -rf ipipe_build.sh output/
    cp -rf requirements.txt output/
    cp -rf pyproject.toml output/
    cp -rf README.md output/
}
build_image() {
    if [ -n "$1" ] && [ "$1" != "" ]
    then
        image=ccr-2dgcnf1d-vpc.cnc.bj.baidubce.com/famou/v2:$1
    else
        image=ccr-2dgcnf1d-vpc.cnc.bj.baidubce.com/famou/v2:$(date +"%Y%m%d-%H%M%S")
    fi
    echo "image: ${image} building..."
    docker build -t ${image} -f Dockerfile_ipipe .
    docker push ${image}
    echo "image: ${image} build success!"
    echo "IMAGE=${image}" >> $WORKSPACE/AGILE_OUT
}


if [[ "${CI_STAGE}" == "compile" ]]; then
prepare_output
fi

if [[ "${CI_STAGE}" == "build_image" ]]; then
build_image ${IMAGE_TAG}
fi
