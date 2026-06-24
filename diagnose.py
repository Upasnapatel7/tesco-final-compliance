import sys
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
    try:
        __import__(m)
        print(f'OK:   {m}')
    except Exception as e:
        print(f'FAIL: {m}  |  {e}')
