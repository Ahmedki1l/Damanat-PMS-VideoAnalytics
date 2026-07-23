#!/usr/bin/env bash
#
# Build a new pms-video-analytics image, ship it to the server, and roll it out via helm.
#
# Usage:
#   ./deploy.sh 1118
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config — adjust to match your environment
# ---------------------------------------------------------------------------
VERSION="${1:?Usage: $0 <version>   e.g. $0 1118}"

IMAGE_NAME="damanat-pms-video-analytics"
TAR_PATH="../${IMAGE_NAME}-${VERSION}.tar"

REMOTE_USER="kloudspot"
REMOTE_HOST="smart.damanat.com.sa"
REMOTE_DIR="pms"                                   # relative to remote $HOME
REMOTE_TAR="${REMOTE_DIR}/${IMAGE_NAME}-${VERSION}.tar"

# values.yaml lives at a fixed absolute path on the remote server.
REMOTE_VALUES_YAML="/etc/kloudspot/values.yaml"

POST_UPGRADE_SCRIPT="./pms/annotate-ingress_create-pv.sh"

# ---------------------------------------------------------------------------
# 1. Local: pull latest code
# ---------------------------------------------------------------------------
echo "==> git pull"
git pull

# ---------------------------------------------------------------------------
# 2. Local: build image and export as tar
# ---------------------------------------------------------------------------
echo "==> building image ${IMAGE_NAME}:${VERSION}"
docker buildx build \
  --platform linux/amd64 \
  -t "${IMAGE_NAME}:${VERSION}" \
  --output type=docker,dest="${TAR_PATH}" \
  .

# ---------------------------------------------------------------------------
# 3. Local: ship tar to server
# ---------------------------------------------------------------------------
echo "==> scp ${TAR_PATH} to ${REMOTE_HOST}:~/${REMOTE_DIR}"
scp "${TAR_PATH}" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_DIR}/"

# ---------------------------------------------------------------------------
# 4. Remote: import image, bump values.yaml, helm upgrade, post-upgrade script
#    Uses `bash -ic` so your existing aliases (mimp, kloudspot) are available.
# ---------------------------------------------------------------------------
echo "==> running remote deployment steps"
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" bash -ic "'
  set -euo pipefail

  echo \"--> importing image\"
  echo \"kloudspot123\" | sudo -S -p \"\" -v
  mimp ${REMOTE_TAR}

  echo \"--> updating values.yaml to version ${VERSION}\"
  sed -i \"s/version: \\\"[0-9]*\\\" #VA/version: \\\"${VERSION}\\\" #VA/\" ${REMOTE_VALUES_YAML}

  echo \"--> helm upgrade\"
  kloudspot upgrade

  echo \"--> running post-upgrade script\"
  ${POST_UPGRADE_SCRIPT}
'"

echo "==> done: ${IMAGE_NAME}:${VERSION} deployed"
