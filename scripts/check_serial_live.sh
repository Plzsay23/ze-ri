#!/usr/bin/env bash

while true; do
  clear
  echo "===== TIME ====="
  date

  echo
  echo "===== USB DEVICES ====="
  lsusb | grep -Ei "10c4|1a86|cp210|ch340|ch341|ch343|qin|arduino|silicon" || true

  echo
  echo "===== TTY DEVICES ====="
  ls -lah /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "no ttyUSB/ttyACM"

  echo
  echo "===== /dev/serial/by-id ====="
  ls -lah /dev/serial/by-id 2>/dev/null || echo "no by-id"

  echo
  echo "===== /dev/serial/by-path ====="
  ls -lah /dev/serial/by-path 2>/dev/null || echo "no by-path"

  echo
  echo "===== CURRENT FIXED LINKS ====="
  ls -lah /dev/ydlidar_nb /dev/arduino_nb 2>/dev/null || echo "no fixed links"

  sleep 1
done
