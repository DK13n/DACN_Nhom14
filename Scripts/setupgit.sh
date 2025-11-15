#!/bin/bash
set -e

ssh-keygen -t ed25519 -C "Thanh2005.hs@gmail.com"
cat ~/.ssh/id_ed25519.pub


git remote add origin git@github.com:DK13n/DACN_Nhom14.git

git clone -b ThanhVV --single-branch https://github.com/DK13n/DACN_Nhom14.git
