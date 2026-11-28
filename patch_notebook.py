"""Patch SUNDAY_AI_Training.ipynb with Drive safety fixes."""
import json, copy

nb = json.load(open('notebooks/SUNDAY_AI_Training.ipynb'))
cells = nb['cells']

# ── Fix 1: Cell 2 — Replace Drive mount cell with verified mount + guard ────
CELL2_OLD = ''.join(cells[2]['source'])
CELL2_NEW = '''\
# --- 1a. Mount Google Drive (for saving checkpoints) ---
import os, subprocess

from google.colab import drive
drive.mount('/content/drive')

# ── Verify Drive is actually mounted ──────────────────────────────────────────
DRIVE_ROOT = '/content/drive/MyDrive'
if not os.path.exists(DRIVE_ROOT):
    raise RuntimeError(
        "\\n❌ Google Drive did NOT mount correctly!\\n"
        "   Run Runtime → Reconnect and re-run this cell before training."
    )

# Create SUNDAY_AI folder in Drive
DRIVE_DIR = f'{DRIVE_ROOT}/SUNDAY_AI'
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)

# ── Sanity-write: confirm we can actually write to Drive ──────────────────────
_test_file = f'{DRIVE_DIR}/.write_test'
try:
    with open(_test_file, 'w') as _f:
        _f.write('ok')
    os.remove(_test_file)
    print(f'✅ Drive mounted and writable. Checkpoints → {DRIVE_DIR}/checkpoints')
except Exception as _e:
    raise RuntimeError(f"\\n❌ Drive is mounted but NOT writable: {_e}")
'''

if CELL2_OLD.strip() != CELL2_NEW.strip():
    cells[2]['source'] = CELL2_NEW
    print("✅ Cell 2 (Drive mount) patched")
else:
    print("⏭  Cell 2 already patched")

# ── Fix 2: Cell 17 — reduce save_interval 1000→200, val_interval 500→200 ────
src17 = ''.join(cells[17]['source'])

src17 = src17.replace(
    "'save_interval': 1000,",
    "'save_interval': 200,   # Save every 200 steps (was 1000) — critical for Drive reliability"
)
src17 = src17.replace(
    "'val_interval': 500,",
    "'val_interval': 200,    # Validate every 200 steps (was 500)"
)

# Add Drive guard just before Trainer init
TRAINER_INIT_MARKER = "# --- Initialize Trainer ---"
DRIVE_GUARD = """\
# ── Guard: confirm Drive is still mounted before training ────────────────────
import os as _os
if not _os.path.exists('/content/drive/MyDrive'):
    raise RuntimeError(
        "\\n❌ Google Drive is NOT mounted!\\n"
        "   Re-run Cell 1 (the Drive mount cell) before training."
    )
_ckpt_dir = TRAINING_CONFIG['save_dir']
_os.makedirs(_ckpt_dir, exist_ok=True)
_test = f'{_ckpt_dir}/.write_test'
try:
    open(_test, 'w').close(); _os.remove(_test)
    print(f"✅ Drive writable — checkpoints will land in: {_ckpt_dir}")
except Exception as _e:
    raise RuntimeError(f"❌ Cannot write to Drive checkpoint dir: {_e}")

"""
if DRIVE_GUARD not in src17:
    src17 = src17.replace(TRAINER_INIT_MARKER, DRIVE_GUARD + TRAINER_INIT_MARKER)
    print("✅ Cell 17 (Trainer config) Drive guard added")
else:
    print("⏭  Cell 17 Drive guard already present")

cells[17]['source'] = src17
print("✅ Cell 17 (Trainer config) intervals patched")

# ── Fix 3: Cell 27 — add os.sync() after final torch.save ───────────────────
src27 = ''.join(cells[27]['source'])
OLD_FINAL = "print(f'✅ Final model saved')"
NEW_FINAL = """\
# Force Drive FUSE buffer to flush — prevents data loss on runtime disconnect
import sys
if sys.platform != 'win32':
    os.sync()
print(f'✅ Final model saved + Drive synced')"""

if OLD_FINAL in src27 and 'os.sync()' not in src27:
    src27 = src27.replace(OLD_FINAL, NEW_FINAL)
    cells[27]['source'] = src27
    print("✅ Cell 27 (final save) sync added")
else:
    print("⏭  Cell 27 already has sync or was not found")

# ── Write patched notebook ───────────────────────────────────────────────────
with open('notebooks/SUNDAY_AI_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n🎉 Notebook patched successfully!")
