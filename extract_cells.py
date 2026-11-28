import json

nb = json.load(open('notebooks/SUNDAY_AI_Training.ipynb'))
keywords = ['drive.mount', 'DRIVE_DIR', 'save_dir', 'checkpoint_every', 
            'checkpoint_dir', 'save_checkpoint', 'best_model', 'Trainer(']

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(k in src for k in keywords):
        print(f'=== CELL {i} ({cell["cell_type"]}) ===')
        print(src[:3000])
        print()
