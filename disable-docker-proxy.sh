#!/bin/bash
sudo rm -f /etc/systemd/system/docker.service.d/http-proxy.conf
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart docker

