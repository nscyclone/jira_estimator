#!/usr/bin/env python
"""Seed ~20 synthetic feedback rows for demo purposes.

Simulates realistic team usage on a Russian medical information system (МИС)
backlog: predicted vs actual days across different modules and subsystems.

Usage:
    python seed_synthetic_feedback.py
    python seed_synthetic_feedback.py --db path/to/custom.db
    python seed_synthetic_feedback.py --clear   # wipe synthetic rows first
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from feedback import init_db

SYNTHETIC_ROWS = [
    {
        "summary": "Реализовать выгрузку СЭМД протокола осмотра врача",
        "description": "Добавить формирование структурированного электронного документа по форме CDA R2. Шаблон согласован с главным врачом.",
        "region": "БАЗОВЫЙ",
        "subsystem": "СЭМД/Выгрузка",
        "commitments": "ТЗР",
        "predicted_days": 3.0,
        "actual_days": 5.0,
    },
    {
        "summary": "Добавить проверку дублей пациентов при регистрации",
        "description": "При создании нового пациента проверять совпадение ФИО + дата рождения + СНИЛС. Показывать предупреждение с найденными совпадениями.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Регистратура/Картотека",
        "commitments": "SLA",
        "predicted_days": 2.0,
        "actual_days": 2.0,
    },
    {
        "summary": "Исправить ошибку расчёта возраста пациента в форме ТАП",
        "description": "Возраст считается неверно для пациентов, рождённых 29 февраля. Приводит к ошибке при открытии ТАП.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Поликлиника/ТАП",
        "commitments": "SLA",
        "predicted_days": 0.5,
        "actual_days": 1.0,
    },
    {
        "summary": "Реализовать модуль автоматического расписания врачей",
        "description": "Разработать алгоритм генерации расписания с учётом ставок, дней отдыха и производственного календаря.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Регистратура/Расписание",
        "commitments": "ТЗР",
        "predicted_days": 5.0,
        "actual_days": 9.0,
    },
    {
        "summary": "Добавить печать направления на госпитализацию",
        "description": "",
        "region": "БАЗОВЫЙ",
        "subsystem": "Стационар/Госпитализация",
        "commitments": "SLA",
        "predicted_days": 1.0,
        "actual_days": 1.0,
    },
    {
        "summary": "Реализовать интеграцию с ФРМО для актуализации реестра МО",
        "description": "Настроить обмен данными с Федеральным реестром медицинских организаций через REST API ЕГИСЗ.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Интеграции/ЕГИСЗ",
        "commitments": "ТЗР",
        "predicted_days": 4.0,
        "actual_days": 7.0,
    },
    {
        "summary": "Перевести модуль лаборатории на новый протокол HL7 FHIR",
        "description": "Текущий обмен по ASTM устарел. Переход на FHIR R4, обновить маппинг результатов анализов.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Лаборатория/ЛИС",
        "commitments": "ТЗР",
        "predicted_days": 5.0,
        "actual_days": 8.0,
    },
    {
        "summary": "Добавить автотесты для сервиса назначений",
        "description": "Покрыть основные сценарии: создание, отмена, исполнение назначения. Использовать JUnit 5 + Mockito.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Стационар/Назначения",
        "commitments": "SLA",
        "predicted_days": 2.0,
        "actual_days": 3.0,
    },
    {
        "summary": "Оптимизировать запрос получения журнала событий пациента",
        "description": "P99 на эндпоинте /patient/events достигает 3 секунд. Добавить индекс по patient_id + created_at, кэшировать агрегаты.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Платформа/БД",
        "commitments": "SLA",
        "predicted_days": 2.0,
        "actual_days": 1.5,
    },
    {
        "summary": "Реализовать отчёт по диспансерному наблюдению",
        "description": "",
        "region": "БАЗОВЫЙ",
        "subsystem": "Аналитика/Отчёты",
        "commitments": "ТЗР",
        "predicted_days": 3.0,
        "actual_days": 3.0,
    },
    {
        "summary": "Устранить утечку памяти в сервисе обработки входящих HL7-сообщений",
        "description": "После 8 часов работы сервис потребляет 4ГБ. Добавить профилировщик, найти объекты без освобождения.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Интеграции/HL7",
        "commitments": "SLA",
        "predicted_days": 2.0,
        "actual_days": 5.0,
    },
    {
        "summary": "Добавить фильтрацию по МКБ-10 в списке случаев лечения",
        "description": "Пользователи запрашивают фильтр по коду диагноза. Поддержать поиск по части кода и по наименованию.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Поликлиника/Случаи",
        "commitments": "SLA",
        "predicted_days": 1.0,
        "actual_days": 1.0,
    },
    {
        "summary": "Обновить зависимости: spring-boot 2.7 → 3.2",
        "description": "Мажорный апгрейд. Обновить конфигурацию security, javax → jakarta, проверить совместимость.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Платформа/Инфраструктура",
        "commitments": "ТЗР",
        "predicted_days": 3.0,
        "actual_days": 5.0,
    },
    {
        "summary": "Разработать разграничение прав доступа по ролям в модуле АПУ",
        "description": "Реализовать RBAC: роли регистратор, врач, главврач, администратор. Описать матрицу доступа в ТЗ.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Платформа/Безопасность",
        "commitments": "ТЗР",
        "predicted_days": 4.0,
        "actual_days": 6.0,
    },
    {
        "summary": "Реализовать push-уведомления о готовности результатов анализов",
        "description": "Отправка уведомления пациенту в мобильное приложение при получении результата из ЛИС.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Уведомления/Мобайл",
        "commitments": "ТЗР",
        "predicted_days": 3.0,
        "actual_days": 4.0,
    },
    {
        "summary": "Написать ТЗ на модуль телемедицины",
        "description": "Проработать требования к видеоконсультациям, записи, хранению медиа, интеграции с расписанием.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Аналитика/Проектирование",
        "commitments": "ТЗР",
        "predicted_days": 3.0,
        "actual_days": 2.0,
    },
    {
        "summary": "Настроить CI/CD пайплайн для микросервиса направлений",
        "description": "GitLab CI: сборка, прогон тестов, статический анализ, деплой на стейджинг.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Платформа/DevOps",
        "commitments": "SLA",
        "predicted_days": 1.5,
        "actual_days": 2.0,
    },
    {
        "summary": "Реализовать повторную отправку СЭМД при ошибке регистрации в РЭМД",
        "description": "Добавить очередь повторных попыток с экспоненциальным откатом. Dead-letter очередь после 5 попыток.",
        "region": "БАЗОВЫЙ",
        "subsystem": "СЭМД/Интеграции",
        "commitments": "SLA",
        "predicted_days": 2.0,
        "actual_days": 3.0,
    },
    {
        "summary": "Провести ревью SQL-запросов модуля отчётности на N+1",
        "description": "Проанализировать все ORM-запросы в отчётном модуле, устранить N+1 с помощью join/fetch.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Аналитика/Отчёты",
        "commitments": "SLA",
        "predicted_days": 1.0,
        "actual_days": 1.5,
    },
    {
        "summary": "Перевести стейджинг-окружение на Kubernetes",
        "description": "Мигрировать со старого docker-compose на k8s. Написать Helm-чарты для всех сервисов.",
        "region": "БАЗОВЫЙ",
        "subsystem": "Платформа/DevOps",
        "commitments": "ТЗР",
        "predicted_days": 5.0,
        "actual_days": 10.0,
    },
]


def seed_synthetic(db_path: str, clear: bool = False) -> None:
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        if clear:
            conn.execute("DELETE FROM feedback WHERE summary IN ({})".format(
                ",".join("?" * len(SYNTHETIC_ROWS))
            ), [r["summary"] for r in SYNTHETIC_ROWS])
            conn.commit()
            print(f"Cleared synthetic rows from {db_path}.")

        existing = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if existing > 0 and not clear:
            print(f"feedback table already has {existing} row(s). Skipping to avoid duplicates.")
            print("Use --clear to wipe synthetic rows first.")
            return

        ts = "2026-06-01T10:00:00+00:00"
        for row in SYNTHETIC_ROWS:
            predicted = row["predicted_days"]
            actual = row["actual_days"]
            delta_pct = abs(predicted - actual) / max(actual, 0.001) * 100

            conn.execute(
                """
                INSERT INTO feedback
                    (created_at, summary, description, region, subsystem, commitments,
                     predicted_days, actual_days, delta_pct, is_used_for_training)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    ts,
                    row["summary"],
                    row["description"],
                    row["region"],
                    row["subsystem"],
                    row["commitments"],
                    predicted,
                    actual,
                    round(delta_pct, 2),
                ),
            )
        conn.commit()

    print(f"Seeded {len(SYNTHETIC_ROWS)} synthetic feedback rows into {db_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=CONFIG["feedback_db_path"])
    parser.add_argument("--clear", action="store_true", help="Delete existing synthetic rows before inserting")
    args = parser.parse_args()
    seed_synthetic(args.db, clear=args.clear)


if __name__ == "__main__":
    main()
