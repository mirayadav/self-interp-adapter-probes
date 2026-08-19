#!/usr/bin/env bash
H=47.47.180.66; P=18638; K=~/.ssh/id_ed25519
case "${1:-}" in
  scp) shift; exec scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $K -P $P "$@" ;;
  *)   exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $K -p $P root@$H "$@" ;;
esac
