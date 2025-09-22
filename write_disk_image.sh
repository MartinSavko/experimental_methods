#!/bin/bash
# https://wiki.gentoo.org/wiki/Handbook:AMD64/Installation/Media
dd if=livegui-amd64-20240915T163400Z.iso of=/dev/sdg bs=4096 status=progress && sync
