# Overnight setup: G14 host + Macs, driven by Claude over SSH

Goal: after landing, leave the G14 and the Macs running all night so Claude (running on the G14) can pull code,
start and stop workers, pre-load model shards and run measurements on every machine, with nobody at the keyboards.
Target model for the night: Qwen3-235B-A22B-Instruct-2507 UD-Q2_K_XL (82.7 GiB, two shards), with gpt-oss-120b
(59 GiB, already proven) as the baseline.

## What to bring

| Machine | Role | Needs |
| --- | --- | --- |
| G14 (Windows) | host, runs Claude, holds the model files | charger, lid open |
| MacBook A (16 GB, about 200 GB free) | peer | charger, lid open |
| MacBook B (16 GB, about 200 GB free) | peer | charger, lid open |
| Azim's Mac (16 GB, about 60 GB free) | peer (and Mac host tests tomorrow) | charger, lid open, a separate user account (below) |
| A hotspot or travel router | private network | must allow device-to-device traffic (no client isolation) |

## On the G14 (once)

1. Settings > System > Power: screen and sleep on power: Never. Lid close action on power: Do nothing.
2. Keep the charger in the barrel port (USB-C cannot keep up under load; last night it dropped to battery).
3. Open Tailscale and log in to your tailnet.
4. Join the hotspot or router. When Windows asks about the network, choose Private.
5. Leave the Claude Code desktop app open on this session and send the "go overnight" message.

## On each Mac (about 15 minutes each)

Do these in order. On Azim's personal Mac do everything in a new account so nothing personal is reachable.

1. **Separate account (Azim's Mac; recommended on all):** System Settings > Users & Groups > Add User: a Standard
   user named `hypercompute`. Log in to it once so its home folder exists, then switch back to the normal account
   (processes keep running for SSH sessions).
2. **Power:** System Settings > Battery > Options: "Prevent automatic sleeping on power adapter when the display is
   off" on. Keep the lid open and the charger in.
3. **Remote Login:** System Settings > General > Sharing > Remote Login on, "Allow access for: Only these users":
   `hypercompute`.
4. **Claude's SSH key:** in Terminal as `hypercompute`:
   ```bash
   mkdir -p ~/.ssh && chmod 700 ~/.ssh
   echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDAmzgir9IcbEAzGZ58M3Znzss0eMaPaWtPW2UuGH8JX hypercompute-g14-overnight' >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```
5. **Tailscale:** install from https://tailscale.com/download (works without admin via the App Store build if the
   account can install apps; otherwise do it from the admin account), log in to the same tailnet, and send Claude the
   name from `tailscale status`.
6. **Tools and code** (as `hypercompute`):
   ```bash
   xcode-select --install          # git and clang (clang lets Claude build the Mac extension overnight)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Open a new Terminal, then:
   ```bash
   git clone https://github.com/amemon12/Hypercompute.git hypercompute && cd hypercompute
   git checkout codex/iroh-transport
   uv run hypercompute engine install --yes
   uv run hypercompute doctor
   ```
   The repository is private: the account cloning it needs GitHub access (`gh auth login`, or a personal access token
   when git asks for a password).
7. **Join the same hotspot or router as the G14.**

That is all. Claude does the joining, worker settings, shard downloads and runs over SSH.

## What Claude will and will not do on the Macs

- Will: `git pull` in `~/hypercompute`, start or stop `hypercompute join` with `--managed --max-disk` limits, run
  `caffeinate`, download model shards into the account's RPC cache, build the Mac worker extension with clang, read
  logs, and report measurements.
- Will not: use `sudo`, touch any other account or folder, or change system settings.
- Disk: shards stay under the `--max-disk` cap and at least 20 GiB free is always kept. Azim's Mac gets a small cap.

## Undo afterwards

On each Mac: Remote Login off, delete `~/.ssh/authorized_keys` in the `hypercompute` account (or delete the whole
account, which also removes `~/hypercompute`, `~/.hypercompute` and the RPC cache), and remove the Mac from the
tailnet. On the G14: `ssh-keygen` key `~/.ssh/hypercompute_ed25519` can be deleted.
