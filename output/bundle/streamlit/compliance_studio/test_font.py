import sys
sys.path.insert(0, '.')
import traceback
from PIL import Image
import numpy as np

# Patch render_creative to show the real error
import creative_renderer as cr

img = Image.fromarray((np.ones((200,200,3))*180).astype('uint8'))

try:
    result = cr.render_creative(
        dimensions=(600,600),
        packshots=[img.convert('RGBA')],
        headline='BIG TEXT TEST',
        subhead='small subhead',
        brand_name='Tesco',
        font_name='Lora (Elegant Serif)',
        font_weight='bold',
        headline_size=100,
        subhead_size=40,
        bg_color='#BFE0F5',
        badge_show=False,
    )
    print('Result:', result)
except Exception as e:
    print('CRASH:', e)
    traceback.print_exc()

# Also test each section manually
print()
print('Testing _make_bg...')
try:
    bg = cr._make_bg(600, 600, '#BFE0F5', '')
    print('_make_bg OK:', bg.size)
except Exception as e:
    print('_make_bg FAILED:', e)
    traceback.print_exc()

print()
print('Testing _load_font at size 100...')
try:
    f = cr._load_font('Lora (Elegant Serif)', 100, 'bold')
    print('Font OK:', f)
except Exception as e:
    print('Font FAILED:', e)
    traceback.print_exc()

print()
print('Checking render_creative source for return statement...')
import inspect
src = inspect.getsource(cr.render_creative)
if 'return canvas' in src:
    print('return canvas found in source')
else:
    print('WARNING: no return statement found!')
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'return' in line:
        print(f'  line {i}: {line}')