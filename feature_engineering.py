DEV_WORDS = ['разраб', 'dev', 'implement', 'feature', 'фич', 'кодиров']
TEST_WORDS = ['тест', 'test', 'qa', 'проверк', 'autotest', 'автотест']
ANALYSIS_WORDS = ['анализ', 'anali', 'тз', 'требован', 'проектир', 'requirement']


def compute_text_features(full_text: str, description: str) -> dict:
    text_lower = full_text.lower()
    return {
        'has_description': int(bool(description.strip())),
        'has_code_block': int('{code' in description.lower()),
        'is_dev_task': int(any(w in text_lower for w in DEV_WORDS)),
        'is_test_task': int(any(w in text_lower for w in TEST_WORDS)),
        'is_analysis_task': int(any(w in text_lower for w in ANALYSIS_WORDS)),
        'text_len': len(full_text),
        'word_count': len(full_text.split()),
    }
