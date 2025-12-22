#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p log

nohup bash 2000.sh > log/2000.log 2>&1 &
nohup bash 2001.sh > log/2001.log 2>&1 &
nohup bash 2002.sh > log/2002.log 2>&1 &
nohup bash 2003.sh > log/2003.log 2>&1 &
nohup bash 2004.sh > log/2004.log 2>&1 &
