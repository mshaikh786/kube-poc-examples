#!/bin/bash

APPS=
REFERENCE='10.10.10.221'
WORKSPACE='10.10.10.152'
SW='10.10.10.171'



#mkdir -p /mnt/apps
mkdir -p /mnt/workspace
mkdir -p /mnt/reference
mkdir -p /mnt/sw


#mount 10.10.10.207:/apps /mnt/apps
mount ${REFERENCE}:/reference /mnt/reference
mount ${WORKSPACE}:/workspace /mnt/workspace
mount ${SW}:/sw /mnt/sw
