#!/usr/bin/env bash
# SSH wrapper for the RunPod pod. Port changes on every restart — update here only.
POD_HOST=69.30.85.29
POD_PORT=22179
POD_KEY=~/.ssh/id_ed25519
case "${1:-}" in
  scp) shift; exec scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $POD_KEY -P $POD_PORT "$@" ;;
  *)   exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $POD_KEY -p $POD_PORT root@$POD_HOST "$@" ;;
esac
