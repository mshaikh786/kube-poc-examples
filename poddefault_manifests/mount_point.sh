#!/bin/bash
mkdir -p /mnt/workspace
mkdir -p /mnt/apps
mkdir -p /mnt/reference
mkdir -p /mnt/sw


mount 10.10.10.157:/workspace /mnt/workspace
mount 10.10.10.190:/sw /mnt/sw
mount 10.10.10.207:/apps /mnt/apps
mount 10.10.10.211:/reference /mnt/reference
