# Entry V2 Docker deployment

This bundle injects runtime settings and mounts production configuration without
baking credentials or camera passwords into the image.

It must be the only deployment manifest that owns the
`pms-video-analytics` container. If Jenkins or another Compose project already
owns that container, copy the `env_file`, `volumes`, `network`, and healthcheck
settings into that existing manifest instead of starting a second owner.

## 1. Preserve the current gallery

Before changing mounts, inspect the running container:

```bash
docker inspect pms-video-analytics --format '{{.Config.Image}}'
docker inspect pms-video-analytics \
  --format '{{range .Mounts}}{{println .Type .Source "->" .Destination}}{{end}}'
```

If `/app/vehicle_images` is already a durable bind mount, reuse its source as
`VA_GALLERY_DIR`. If it is not mounted (or uses storage that the new manifest
will not reuse), copy the existing history during a controlled maintenance
stop before the first recreation:

```bash
va_deploy_user=$(id -un)
va_deploy_group=$(id -gn)
sudo install -d -o "$va_deploy_user" -g "$va_deploy_group" -m 0750 \
  /var/lib/jenkins/data/pms-video-analytics/vehicle_images
docker stop pms-video-analytics
sudo docker cp pms-video-analytics:/app/vehicle_images/. \
  /var/lib/jenkins/data/pms-video-analytics/vehicle_images/
```

If this new Compose bundle is taking ownership from an unmanaged old container,
rename that stopped container before `up` so it remains available for rollback:

```bash
docker rename pms-video-analytics pms-video-analytics-before-entry-v2
```

When Jenkins or an existing Compose project remains the owner, do not rename the
container; add the settings from this bundle to that owner and let it recreate
the service.

## 2. Prepare external configuration

On the Docker host:

```bash
va_deploy_user=$(id -un)
va_deploy_group=$(id -gn)
sudo install -d -o "$va_deploy_user" -g "$va_deploy_group" -m 0750 \
  /var/lib/jenkins/configs/pms-video-analytics
sudo install -d -o "$va_deploy_user" -g "$va_deploy_group" -m 0750 \
  /var/lib/jenkins/data/pms-video-analytics/vehicle_images
# First installation only. The guards preserve live edited files on later runs.
if [ ! -e /var/lib/jenkins/configs/pms-video-analytics/compose.env ]; then
  sudo install -o "$va_deploy_user" -g "$va_deploy_group" -m 0600 \
    deploy/entry-v2/compose.env.example \
    /var/lib/jenkins/configs/pms-video-analytics/compose.env
fi
if [ ! -e /var/lib/jenkins/configs/pms-video-analytics/va.env ]; then
  sudo install -o "$va_deploy_user" -g "$va_deploy_group" -m 0600 \
    deploy/entry-v2/va.env.example \
    /var/lib/jenkins/configs/pms-video-analytics/va.env
fi
if [ ! -e /var/lib/jenkins/configs/pms-video-analytics/config.yaml ]; then
  sudo install -o "$va_deploy_user" -g "$va_deploy_group" -m 0600 \
    /path/to/completed/config.entry-v2.yaml \
    /var/lib/jenkins/configs/pms-video-analytics/config.yaml
fi
```

Run Compose as that same deployment user. If Jenkins runs Compose instead,
replace the owner/group above with the Jenkins service account so it can read
`va.env`; do not solve this by making the secret files world-readable.

Find the shared network before editing (the current image was recorded before
any optional container rename in step 1):

```bash
docker inspect pms-ai \
  --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}'
docker inspect pms-mssql \
  --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}'
sudoedit /var/lib/jenkins/configs/pms-video-analytics/compose.env
sudoedit /var/lib/jenkins/configs/pms-video-analytics/va.env
```

Edit `compose.env` and set:

- `VA_IMAGE` to the immutable registry tag/digest being deployed.
- `DAMANAT_DOCKER_NETWORK` to the existing network containing both `pms-ai` and
  `pms-mssql`.
- Keep `VA_BIND_IP=127.0.0.1` unless a protected reverse proxy or firewall must
  reach VA through the host port. PMS reaches VA on the Docker network and does
  not need a public host bind.
- Host paths only if the defaults do not match the server.

Edit `va.env` and set `ENTRY_V2_SERVICE_KEY` to the same 64-hex-character
secret used by PMS. Generate one with `openssl rand -hex 32`. Keep
`ENTRY_V2_MODE=off` for the first deployment.

Edit `/var/lib/jenkins/configs/ps-ai/.env` and ensure PMS has the matching
initial values (the URL must be plain text, not Markdown link syntax):

```dotenv
PMS_API_URL=http://pms-video-analytics:8000
ENTRY_V2_MODE=off
ENTRY_V2_SERVICE_KEY=<same-64-hex-secret>
```

The safe VA template initially accepts CAM-23's local `Park_Entry` event. After
capturing a real CAM-23 webhook, append its exact raw values while retaining the
local aliases:

```dotenv
ENTRY_V2_PRIMARY_LINES=<measured-hikvision-line>,Park_Entry
ENTRY_V2_PRIMARY_DIRECTIONS=<measured-hikvision-inward-direction>,ramp-entry
```

## 3. Validate and deploy

From the VA repository checkout on the Docker host:

```bash
docker compose \
  --env-file /var/lib/jenkins/configs/pms-video-analytics/compose.env \
  -f deploy/entry-v2/docker-compose.yml \
  config --quiet

docker compose \
  --env-file /var/lib/jenkins/configs/pms-video-analytics/compose.env \
  -f deploy/entry-v2/docker-compose.yml \
  pull

docker compose \
  --env-file /var/lib/jenkins/configs/pms-video-analytics/compose.env \
  -f deploy/entry-v2/docker-compose.yml \
  up -d --force-recreate
```

For hosts that provide the standalone command, replace `docker compose` with
`docker-compose`.

Editing `/var/lib/jenkins/configs/ps-ai/.env` does not update the running PMS
process. Recreate `pms-ai` through the Jenkins job or existing Compose manifest
that owns it. A plain `docker restart pms-ai` is sufficient only when that exact
external file is already mounted into the container.

## 4. Verify off mode

```bash
docker inspect pms-video-analytics \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
docker exec pms-video-analytics sh -lc \
  'printf "entry=%s motion=%s state=%s single=%s processes=%s\n" \
  "$ENTRY_V2_MODE" "$VA_MOTION_SCHEDULER_MODE" "$VA_SLOT_STATE_MODE" \
  "$VA_SINGLE_PROCESS" "$VA_PROCESS_COUNT"'
docker exec pms-video-analytics \
  curl -fsS http://127.0.0.1:8000/api/health
docker exec pms-ai python -c \
  'from app.config import settings; print(settings.PMS_API_URL, settings.ENTRY_V2_MODE, bool(settings.ENTRY_V2_SERVICE_KEY))'
docker logs --tail 200 pms-video-analytics
```

Require the config and gallery mounts, `ENTRY_V2_MODE=off`, both policy modes
`shadow`, one process, and healthy engine/model/database/camera status.

The PMS check must print the plain VA URL, `off`, and `True` for the configured
service key.

## 5. Enter shadow mode

After the off-mode health gate passes, edit the external `va.env` and PMS `.env`:

```dotenv
ENTRY_V2_MODE=shadow
```

Run the VA `up -d --force-recreate` command again, then immediately recreate PMS
through its current Jenkins/Compose owner. Editing either file without recreating
its container does not change the running process. Both services must use the
same non-empty service key and report the same Entry V2 mode. Do not promote to
authoritative until the documented labelled shadow and field-calibration gates
pass.

## Rollback

Set both services back to `ENTRY_V2_MODE=shadow` (or `off` before authoritative
has ever been enabled), recreate them, and restore the previous immutable image
tag if needed. Preserve the gallery bind mount. Never use `down -v`.
