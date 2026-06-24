with open('creative_studio.py', 'r', encoding='utf-8') as f:
    src = f.read()
src = src.replace('use_column_width=True', 'use_container_width=True')
src = src.replace('use_column_width=False', 'use_container_width=False')
with open('creative_studio.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('Fixed')