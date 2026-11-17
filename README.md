# BrotatoRL

Brotato as a [Gymnasium](https://gymnasium.farama.org/) environment for reinforcement learning.

A small mod runs inside a private copy of the game and talks to Python over a local socket:
- `reset()` starts a new run;
- `step(action)` moves the character, while the game aims and shoots on its own;
- level-ups and the shop between waves are handled by a swappable "meta policy" (rules, random or learned).

> **No game files are included.** You need your own copy of Brotato on Steam. This project is not
> affiliated with Blobfish or the Brotato publishers.

## Requirements

- **System**: Windows 10 or 11.
- **Game**: Brotato on Steam, **version 1.1.15.4**. The mod relies on game internals. Version 1.2.x is
  untested and may need changes in `bridge.gd`.
- **Python**: 3.10+. For training, also PyTorch and Stable-Baselines3.
- **Optional**: `ffmpeg` on `PATH` for video recording.

## Installation

```powershell
git clone https://github.com/Wojtur28/brotatoRL.git
cd brotatoRL
python -m venv .venv
.venv\Scripts\activate
pip install -e python

# only for training
pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
pip install -e "python[train]"

python python\scripts\setup_lab.py     # copy the game to lab\ and install the mod
python python\scripts\smoke_test.py    # 13 end-to-end checks, about 2 minutes
```

**Finding the game.** `setup_lab.py` finds Brotato in your Steam libraries. If that fails, use
`--game-dir` or set `BROTATO_GAME_DIR`.

**After changing the mod.** Run `setup_lab.py` again after every change to the mod.

## Usage

```python
import brotato_rl, gymnasium as gym

env = gym.make("BrotatoRL-v0", character="character_well_rounded", danger=0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

- **Actions**: `Discrete(9)` by default (stay + 8 directions), or continuous with `action_type="continuous"`.
- **Observation**: a dict with the player's stats and the nearest enemies, projectiles, spawns,
  materials and consumables (`python\brotato_rl\encoding.py`).
- **Reward**: survival, damage taken, materials, waves cleared, death and win
  (`python\brotato_rl\rewards.py`). It can be changed with `RewardConfig`.
- **Between waves**: `RandomMetaPolicy`, `RuleMetaPolicy`, or your own class with `level_up`, `item`
  and `shop` methods (`python\brotato_rl\meta.py`).
- **Parallel games**: `brotato_rl.vector.make_vec_env(n)` (Gymnasium) or
  `brotato_rl.sb3.make_sb3_vec_env(n)` (Stable-Baselines3). Each game runs headless and needs about
  0.45 GB RAM.

Watch the built-in bot, or set everything up in a local web panel:

```powershell
python python\scripts\play.py --bot heuristic --watch
python python\scripts\webapp.py --open
```

## Training

```powershell
# movement agent (PPO): bot demonstrations for the warm start, then training
python python\scripts\collect_demos.py --name bot_melee60 --config python\configs\ppo5.json --episodes 60
python python\scripts\train.py --name my_run --config python\configs\ppo5.json
python python\scripts\train.py --name my_run --resume
python python\scripts\evaluate.py --name my_eval --model runs\my_run\best\model.zip --episodes 96 --seed 10000
```

`python\configs\ppo5.json` is the recommended setup:
- a per-entity (DeepSets) network;
- a behaviour cloning warm start from the bot;
- the bot as a teacher during PPO.

Override any config field with `--set key=value`. Runs are saved in `runs\<name>\`: checkpoints,
`best\`, `final\`, logs and TensorBoard files.

Use separate seed ranges: 10000+ for tests, 20000+ for validation during training, 30000+ for
recordings.

**Shop agent.** It trains on top of a frozen movement model exported to numpy:

```powershell
python -m brotato_rl.pilot runs\my_run\final\model.zip models\pilot\pilot.npz
python python\scripts\train_shop.py collect --name shop_demos --episodes 300 --pilot models\pilot\pilot.npz
python python\scripts\train_shop.py bc --name shop1 --demos demos\shop_demos.npz
python python\scripts\train_shop.py ppo --name shop1 --init runs\shop1\bc\shop.pt --hours 10 --pilot models\pilot\pilot.npz
python python\scripts\train_shop.py stop --name shop1      # stop; continue later with: ppo ... --resume
```

## Watching and recording

```powershell
python python\scripts\play.py --run my_run --at 2.5M --watch          # any checkpoint, in a game window
python python\scripts\record_video.py --run my_run --at best --seed 30000
python python\scripts\play.py --run my_run --at best --replay-dir showcase\replays
python python\scripts\progress.py --runs my_run                        # learning curve: showcase\progress.html
```

To view replays, open `viewer\replay.html` in a browser and drop the `.json.gz` files on it. Every
script has `--help`.

## How it works

- **Mod loading.** The Steam build only loads Workshop mods. `setup_lab.py` therefore copies the game
  to `lab\game` and adds three things: an `override.cfg`, a small mod-loader shim and the built mod.
  The Steam install is only read.
- **The mod.** It lives in `mod\rl_bridge\`. It runs only when started with `--rl-port`, and it
  pauses the game every `frame_skip` frames until Python answers. Messages are length-prefixed JSON
  over `127.0.0.1`.
- **Your saves.** They are never touched. Each instance gets its own data folder under
  `lab\instances\`. The copy refuses to run against the real `%APPDATA%\Brotato`, and progress
  saving is off. The smoke test compares checksums of your saves before and after.
- **Steam.** It can stay closed, since everything works offline. Achievements and stat uploads are
  blocked either way.

## Troubleshooting

- **`No module named 'brotato_rl'`**: activate `.venv` first.
- **`lab is incomplete`**, or mod changes have no effect: run `setup_lab.py`.
- **The game exits before connecting**: the error shows the end of the game log. The full log is in
  `lab\instances\<id>\BrotatoRL\logs\godot.log`.

## License

MIT, see [LICENSE](LICENSE). Brotato and its assets belong to their respective owners.
