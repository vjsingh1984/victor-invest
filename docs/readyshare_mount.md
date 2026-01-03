# ReadySHARE mount helper (macOS)

Steps below mount the ReadySHARE SMB export on `/Volumes/readyshare` and let `launchd` keep it available after login.

## 1. Prepare credentials

1. Create `/etc/readyshare-credentials` as root:
   ```bash
   sudo sh -c 'cat >/etc/readyshare-credentials <<EOF
   user=admin
   password=Rudra@20240912#
   EOF'
   sudo chmod 600 /etc/readyshare-credentials
   ```
2. Adjust `user=` / `password=` if the router creds change.

## 2. Manual mount helper script

1. Make the helper executable:
   ```bash
   chmod +x scripts/mount_readyshare.sh
   ```
2. Mount on demand (defaults shown; override via `READYSHARE_*` env vars or pass a mount point as `$1`):
   ```bash
   scripts/mount_readyshare.sh        # mounts to /Volumes/readyshare
   scripts/mount_readyshare.sh /Volumes/custom  # custom mount target
   ```
   The script creates the mount point if necessary and percent-encodes special characters before calling `mount_smbfs`.

## 3. LaunchAgent for auto-mounting

1. Copy the template and update the absolute paths:
   ```bash
   mkdir -p ~/Library/LaunchAgents
   cp config/macos/com.investigator.readyshare.mount.plist \
      ~/Library/LaunchAgents/
   sed -i '' \
     -e "s#/ABSOLUTE/PATH/TO/InvestiGator#${PWD}#g" \
     ~/Library/LaunchAgents/com.investigator.readyshare.mount.plist
   ```
2. Load the LaunchAgent (runs at login and retries every 5 minutes if unmounted):
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.investigator.readyshare.mount.plist 2>/dev/null || true
   launchctl load -w ~/Library/LaunchAgents/com.investigator.readyshare.mount.plist
   ```
3. Inspect logs when troubleshooting:
   ```bash
   tail -f /tmp/readyshare-mount.log
   ```

Unload with `launchctl unload -w ~/Library/LaunchAgents/com.investigator.readyshare.mount.plist` if you no longer want the mount to persist.
