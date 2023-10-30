#!/bin/bash
time -p python tools/ExtractFeatures.py \
--soccernet_dirpath ${DATA_DIR}/feature_extract/ \
--game_ID 0 \
--back_end=TF2 \
--features=ResNET \
--video LQ \
--transform crop \
--overwrite \
--verbose --GPU ${OMPI_COMM_WORLD_LOCAL_RANK} &> log_${OMPI_COMM_WORLD_LOCAL_RANK}.txt 

