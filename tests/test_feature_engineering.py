import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engineering import compute_text_features


def test_empty_description_no_flag():
    result = compute_text_features("Реализовать кнопку", "")
    assert result["has_description"] == 0


def test_non_empty_description_flagged():
    result = compute_text_features("Реализовать кнопку", "Описание задачи.")
    assert result["has_description"] == 1


def test_code_block_detected():
    result = compute_text_features("Задача", "{code}\nint x = 1;\n{code}")
    assert result["has_code_block"] == 1


def test_no_code_block():
    result = compute_text_features("Задача", "Просто текст без кода")
    assert result["has_code_block"] == 0


def test_dev_task_russian():
    result = compute_text_features("Разработать новый модуль отчётности", "")
    assert result["is_dev_task"] == 1


def test_dev_task_english():
    result = compute_text_features("Implement new feature for auth", "")
    assert result["is_dev_task"] == 1


def test_test_task_keyword():
    result = compute_text_features("Тестирование API авторизации", "")
    assert result["is_test_task"] == 1


def test_analysis_task_keyword():
    result = compute_text_features("Анализ требований к модулю направлений", "")
    assert result["is_analysis_task"] == 1


def test_text_len_and_word_count():
    text = "Одно два три"  # 12 chars, 3 words
    result = compute_text_features(text, "")
    assert result["text_len"] == len(text)
    assert result["word_count"] == 3


def test_neutral_text_no_task_flags():
    result = compute_text_features("Обновить конфигурацию сервера", "Изменить порт на 8080.")
    assert result["is_dev_task"] == 0
    assert result["is_test_task"] == 0
    assert result["is_analysis_task"] == 0


def test_all_expected_keys_present():
    result = compute_text_features("Задача", "Описание")
    assert set(result.keys()) == {
        "has_description", "has_code_block",
        "is_dev_task", "is_test_task", "is_analysis_task",
        "text_len", "word_count",
    }
