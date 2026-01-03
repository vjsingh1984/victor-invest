#!/usr/bin/env bash
set -euo pipefail

# Mounts the ReadySHARE SMB volume on macOS, defaulting to /Volumes/readyshare.
cred_file="${READYSHARE_CREDENTIALS:-/etc/readyshare-credentials}"
mount_point="${1:-/Volumes/readyshare}"
share_host="${READYSHARE_HOST:-192.168.1.1}"
share_name="${READYSHARE_SHARE:-USB_Storage}"

if [[ ! -f "${cred_file}" ]]; then
  echo "Credential file not found: ${cred_file}" >&2
  exit 1
fi

read_value() {
  local key="$1"
  awk -F= -v key="${key}" '$1 == key {print substr($0, index($0,"=")+1)}' "${cred_file}" | tail -n1
}

user="$(read_value "user")"
password="$(read_value "password")"

if [[ -z "${user}" || -z "${password}" ]]; then
  echo "Credential file must include 'user=' and 'password=' lines." >&2
  exit 1
fi

urlencode() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import quote
print(quote(sys.argv[1], safe=''))
PY
}

encoded_user="$(urlencode "${user}")"
encoded_password="$(urlencode "${password}")"
smb_url="//${encoded_user}:${encoded_password}@${share_host}/${share_name}"

if ! mount | grep -q "on ${mount_point} "; then
  if [[ ! -d "${mount_point}" ]]; then
    sudo mkdir -p "${mount_point}"
  fi
  echo "Mounting ${smb_url} to ${mount_point}..."
  sudo mount_smbfs "${smb_url}" "${mount_point}"
else
  echo "${mount_point} is already mounted. Nothing to do."
fi
