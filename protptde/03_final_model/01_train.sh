#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p log

nohup bash 1020.sh > log/1020.log 2>&1 &
nohup bash 1021.sh > log/1021.log 2>&1 &
nohup bash 1022.sh > log/1022.log 2>&1 &
nohup bash 1023.sh > log/1023.log 2>&1 &