import sys
import traceback
sys.path.insert(0, r'c:\Users\upasn\Downloads\tesco_genai_v9\tesco_upgraded')

mods = [
    'brand_config',
    'creative_renderer',
    'ml_pipeline',
    'video_editor',
    'ai_creative_director',
    'auth',
    'creative_studio',
    'advanced_ai_tab',
    'policy_analyser',
    'ai_creative_assistant',
]

for m in mods:
    print(f'Trying: {m} ...', flush=True)
    try:
        __import__(m)
        print(f'  OK:   {m}', flush=True)
    except Exception as e:
        print(f'  FAIL: {m}', flush=True)
        print(f'  ERROR: {e}', flush=True)
        traceback.print_exc()
    print('', flush=True)

print('DONE', flush=True)
