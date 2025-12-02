import os
import sys
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from datetime import datetime
import hashlib
import logging
import argparse
import json
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import yaml

# Безопасная загрузка .env с обработкой ошибок кодировки
try:
    load_dotenv(encoding='utf-8')
except UnicodeDecodeError:
    try:
        # Пробуем альтернативные кодировки
        load_dotenv(encoding='cp1251')  # Для Windows кириллицы
    except:
        try:
            load_dotenv(encoding='latin-1')
        except:
            print("⚠️  Предупреждение: не удалось загрузить .env файл. Проверьте кодировку файла.")
except FileNotFoundError:
    print("⚠️  Файл .env не найден. Убедитесь, что он существует в текущей директории.")
except Exception as e:
    print(f"⚠️  Ошибка загрузки .env: {e}")


# -----------------------------
# КОНФИГУРАЦИЯ
# -----------------------------
@dataclass
class Config:
    """Конфигурация приложения"""
    # OpenAI
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "faq"

    # Обработка
    batch_size: int = 20
    max_workers: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Валидация
    min_question_length: int = 5
    max_question_length: int = 1000
    min_answer_length: int = 10
    max_answer_length: int = 5000

    # Режим работы
    update_mode: str = "upsert"  # upsert, refresh, incremental
    auto_confirm: bool = False
    gui_mode: bool = True

    @classmethod
    def from_env(cls):
        """Создание конфига из переменных окружения"""
        api_key = os.getenv("OPENAI_API_KEY_ORIG")
        if not api_key:
            raise ValueError("❌ Не найден OPENAI_API_KEY_ORIG в .env")

        return cls(
            openai_api_key=api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "faq"),
        )

    @classmethod
    def from_yaml(cls, config_path: str):
        """Загрузка конфига из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Merge с переменными окружения
            api_key = os.getenv("OPENAI_API_KEY_ORIG") or data.get('openai', {}).get('api_key')
            if not api_key:
                raise ValueError("❌ API ключ не найден ни в config.yaml, ни в .env")

            return cls(
                openai_api_key=api_key,
                openai_base_url=data.get('openai', {}).get('base_url',
                                                           os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                openai_model=data.get('openai', {}).get('model', cls.openai_model),
                qdrant_host=data.get('qdrant', {}).get('host', cls.qdrant_host),
                qdrant_port=data.get('qdrant', {}).get('port', cls.qdrant_port),
                collection_name=data.get('qdrant', {}).get('collection', cls.collection_name),
                batch_size=data.get('processing', {}).get('batch_size', cls.batch_size),
                max_workers=data.get('processing', {}).get('max_workers', cls.max_workers),
            )
        except FileNotFoundError:
            logging.warning(f"⚠️  Config файл не найден: {config_path}, используем значения по умолчанию")
            return cls.from_env()
        except Exception as e:
            logging.error(f"❌ Ошибка чтения config.yaml: {e}")
            return cls.from_env()


# -----------------------------
# ЛОГИРОВАНИЕ
# -----------------------------
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Настройка логирования"""
    if log_file is None:
        log_file = f"faq_loader_{datetime.now():%Y%m%d_%H%M%S}.log"

    # Создаем папку для логов если её нет
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger('FAQLoader')
    logger.setLevel(logging.INFO)

    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Хендлер для файла
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Хендлер для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"📝 Логи сохраняются в: {log_path}")

    return logger


# -----------------------------
# УТИЛИТЫ
# -----------------------------
def generate_stable_id(question: str, category: str = "", answer: str = "") -> int:
    """
    Генерация стабильного ID на основе содержимого.
    При одинаковом содержимом всегда будет одинаковый ID.
    """
    content = f"{question.strip().lower()}|{category.strip().lower()}|{answer.strip().lower()[:100]}"
    hash_hex = hashlib.sha256(content.encode('utf-8')).hexdigest()
    # Берем первые 16 символов и конвертируем в int, ограничиваем размер
    return int(hash_hex[:16], 16) % (10 ** 9)


def validate_record(record: Dict, config: Config) -> Tuple[bool, str]:
    """Валидация записи FAQ"""
    question = record.get('q', '').strip()

    # Получаем все возможные варианты ответа
    answer = record.get('a', '').strip()
    short_answer = record.get('short_answer', '').strip()
    full_answer = record.get('full_answer', '').strip()

    # Находим самый длинный непустой ответ для валидации
    all_answers = [a for a in [answer, short_answer, full_answer] if a]

    if not question:
        return False, "Пустой вопрос"

    if not all_answers:
        return False, "Нет ни одного ответа (проверьте столбцы: Ответ/Краткий ответ/Развернутый ответ)"

    # Валидируем самый длинный ответ
    longest_answer = max(all_answers, key=len)

    if len(question) < config.min_question_length:
        return False, f"Вопрос слишком короткий (минимум {config.min_question_length} символов)"

    if len(question) > config.max_question_length:
        return False, f"Вопрос слишком длинный (максимум {config.max_question_length} символов)"

    if len(longest_answer) < config.min_answer_length:
        return False, f"Ответ слишком короткий (минимум {config.min_answer_length} символов)"

    if len(longest_answer) > config.max_answer_length:
        return False, f"Ответ слишком длинный (максимум {config.max_answer_length} символов, текущий: {len(longest_answer)})"

    # Проверка на подозрительный контент
    suspicious_patterns = ['nan', 'null', 'none', 'n/a', '#n/a']
    if question.lower() in suspicious_patterns:
        return False, "Подозрительное содержимое в вопросе (nan/null/none)"

    if all(ans.lower() in suspicious_patterns for ans in all_answers if ans):
        return False, "Подозрительное содержимое в ответах (nan/null/none)"

    return True, ""


# -----------------------------
# РАБОТА С EXCEL
# -----------------------------
class ExcelLoader:
    """Загрузчик данных из Excel"""

    # Маппинг возможных названий столбцов
    COLUMN_MAPPING = {
        'id': ['ID', 'id', 'Идентификатор', '№', 'Номер'],
        'question': ['Вопрос (FAQ)', 'Вопрос', 'question', 'Question', 'FAQ', 'Вопросы'],
        'answer': ['Ответ', 'answer', 'Answer', 'Ответы'],
        'short_answer': ['Краткий ответ', 'Краткий_ответ', 'short_answer', 'Short Answer', 'Кратко'],
        'full_answer': ['Развернутый ответ', 'Развернутый_ответ', 'full_answer', 'Full Answer', 'Подробно',
                        'Полный ответ'],
        'instructions': ['Пошаговая инструкция', 'Инструкция', 'instructions', 'Instructions', 'Шаги', 'Как сделать'],
        'documents': ['Список документов', 'Документы', 'documents', 'Documents', 'Нужные документы'],
        'law': ['Закон и статьи', 'Законы', 'Статьи', 'law', 'Law', 'Правовая база', 'Законодательство'],
        'category': ['Категория', 'category', 'Category', 'Раздел', 'Раздел/Категория', 'Тема'],
        'keywords': ['Ключевые слова', 'keywords', 'Keywords', 'Теги', 'Tags', 'Ключевые_слова'],
        'tags': ['Теги', 'tags', 'Tags', 'Метки'],
        'source': ['Источник / ссылка', 'Источник', 'source', 'Source', 'Ссылка', 'URL'],
        'date': ['Дата актуализации', 'Дата', 'date', 'Date', 'Актуально до', 'Дата обновления']
    }

    def __init__(self, logger: logging.Logger, config: Config):
        self.logger = logger
        self.config = config

    def load(self, file_path: str) -> List[Dict]:
        """Загрузка FAQ из Excel файла"""
        try:
            self.logger.info(f"📖 Чтение файла: {os.path.basename(file_path)}")

            # Читаем файл
            df = pd.read_excel(file_path)
            self.logger.info(f"📊 Загружено {len(df)} строк из файла")
            self.logger.info(f"📋 Столбцы в файле: {list(df.columns)}")

            # Находим реальные имена столбцов
            actual_columns = self._map_columns(df.columns)

            # Проверяем обязательные столбцы
            if 'question' not in actual_columns or 'answer' not in actual_columns:
                missing = []
                if 'question' not in actual_columns:
                    missing.append('Вопрос')
                if 'answer' not in actual_columns:
                    missing.append('Ответ')
                raise ValueError(f"❌ Не найдены обязательные столбцы: {', '.join(missing)}")

            # Обрабатываем данные
            records = []
            stats = {
                'total_rows': len(df),
                'processed': 0,
                'skipped_empty': 0,
                'skipped_invalid': 0,
                'validation_errors': []
            }

            for idx, row in df.iterrows():
                # Пропускаем полностью пустые строки
                if row.isnull().all():
                    stats['skipped_empty'] += 1
                    continue

                # Извлекаем данные
                record = self._extract_record(row, actual_columns, idx, file_path)

                if not record:
                    stats['skipped_empty'] += 1
                    continue

                # Валидация
                is_valid, error_msg = validate_record(record, self.config)
                if not is_valid:
                    stats['skipped_invalid'] += 1
                    stats['validation_errors'].append({
                        'row': idx + 2,
                        'question': record.get('q', '')[:50],
                        'error': error_msg
                    })
                    self.logger.warning(f"⚠️  Строка {idx + 2}: {error_msg}")
                    continue

                records.append(record)
                stats['processed'] += 1

            # Логируем статистику
            self._log_statistics(stats)

            return records

        except Exception as e:
            self.logger.error(f"❌ Ошибка при чтении Excel файла: {e}", exc_info=True)
            raise

    def _map_columns(self, df_columns) -> Dict[str, str]:
        """Маппинг названий столбцов"""
        actual_columns = {}

        for standard_name, possible_names in self.COLUMN_MAPPING.items():
            for possible in possible_names:
                if possible in df_columns:
                    actual_columns[standard_name] = possible
                    self.logger.info(f"   ✅ '{possible}' → '{standard_name}'")
                    break

        return actual_columns

    def _extract_record(self, row, actual_columns: Dict, idx: int, file_path: str) -> Optional[Dict]:
        """Извлечение записи из строки DataFrame"""
        # Получаем вопрос
        question = str(row[actual_columns['question']]).strip() if pd.notna(row[actual_columns['question']]) else ""

        # Пропускаем пустые вопросы
        if not question or question.lower() in ['nan', 'null', 'none', 'n/a']:
            return None

        # Получаем ответы из всех возможных столбцов
        answer = ""
        short_answer = ""
        full_answer = ""

        # Приоритет 1: Развернутый ответ
        if 'full_answer' in actual_columns and pd.notna(row[actual_columns['full_answer']]):
            full_answer = str(row[actual_columns['full_answer']]).strip()
            if full_answer and full_answer.lower() not in ['nan', 'null', 'none', 'n/a']:
                answer = full_answer

        # Приоритет 2: Обычный ответ
        if 'answer' in actual_columns and pd.notna(row[actual_columns['answer']]):
            regular_answer = str(row[actual_columns['answer']]).strip()
            if regular_answer and regular_answer.lower() not in ['nan', 'null', 'none', 'n/a']:
                if not answer:  # Используем только если нет развернутого
                    answer = regular_answer

        # Приоритет 3: Краткий ответ
        if 'short_answer' in actual_columns and pd.notna(row[actual_columns['short_answer']]):
            short_answer = str(row[actual_columns['short_answer']]).strip()
            if short_answer and short_answer.lower() not in ['nan', 'null', 'none', 'n/a']:
                if not answer:  # Используем только если нет других
                    answer = short_answer

        # Если нет ни одного валидного ответа - пропускаем
        if not answer:
            self.logger.debug(f"Строка {idx + 2}: пропущена - нет валидного ответа")
            return None

        # Получаем дополнительные поля
        category = str(row[actual_columns.get('category', '')]).strip() if 'category' in actual_columns and pd.notna(
            row[actual_columns['category']]) else ""

        keywords = str(row[actual_columns.get('keywords', '')]).strip() if 'keywords' in actual_columns and pd.notna(
            row[actual_columns['keywords']]) else ""

        tags = str(row[actual_columns.get('tags', '')]).strip() if 'tags' in actual_columns and pd.notna(
            row[actual_columns['tags']]) else ""

        # Объединяем keywords и tags
        all_keywords = ", ".join(filter(None, [keywords, tags]))

        instructions = str(
            row[actual_columns.get('instructions', '')]).strip() if 'instructions' in actual_columns and pd.notna(
            row[actual_columns['instructions']]) else ""

        documents = str(row[actual_columns.get('documents', '')]).strip() if 'documents' in actual_columns and pd.notna(
            row[actual_columns['documents']]) else ""

        law = str(row[actual_columns.get('law', '')]).strip() if 'law' in actual_columns and pd.notna(
            row[actual_columns['law']]) else ""

        source = str(row[actual_columns.get('source', '')]).strip() if 'source' in actual_columns and pd.notna(
            row[actual_columns['source']]) else ""

        update_date = str(row[actual_columns.get('date', '')]).strip() if 'date' in actual_columns and pd.notna(
            row[actual_columns['date']]) else ""

        # Генерируем стабильный ID
        stable_id = generate_stable_id(question, category, answer)

        # Логируем для отладки (первые 5 записей)
        if idx < 5:
            self.logger.debug(
                f"Строка {idx + 2}: q={question[:50]}... a_len={len(answer)} short={len(short_answer)} full={len(full_answer)}")

        return {
            "id": stable_id,
            "q": question,
            "a": answer,
            "short_answer": short_answer,
            "full_answer": full_answer,
            "instructions": instructions,
            "documents": documents,
            "law": law,
            "category": category,
            "keywords": all_keywords,
            "source": source,
            "update_date": update_date,
            "row_number": idx + 2,
            "source_file": os.path.basename(file_path)
        }

    def _log_statistics(self, stats: Dict):
        """Вывод статистики обработки"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"📊 СТАТИСТИКА ОБРАБОТКИ EXCEL:")
        self.logger.info(f"   • Всего строк: {stats['total_rows']}")
        self.logger.info(f"   • Обработано: {stats['processed']}")
        self.logger.info(f"   • Пропущено (пустые): {stats['skipped_empty']}")
        self.logger.info(f"   • Пропущено (невалидные): {stats['skipped_invalid']}")

        if stats['validation_errors']:
            self.logger.warning(f"\n⚠️  ОШИБКИ ВАЛИДАЦИИ (первые 5):")
            for i, err in enumerate(stats['validation_errors'][:5], 1):
                self.logger.warning(f"   {i}. Строка {err['row']}: {err['error']}")
                self.logger.warning(f"      Вопрос: {err['question']}...")

        self.logger.info(f"{'=' * 60}\n")


# -----------------------------
# РАБОТА С EMBEDDINGS
# -----------------------------
class EmbeddingService:
    """Сервис для создания embeddings через OpenAI API"""

    def __init__(self, logger: logging.Logger, config: Config):
        self.logger = logger
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        self.total_tokens = 0
        self.total_requests = 0

    def get_embedding(self, text: str, retry_count: int = 0) -> Optional[List[float]]:
        """Получение embedding с retry логикой"""
        try:
            start_time = time.time()

            resp = self.client.embeddings.create(
                model=self.config.openai_model,
                input=text
            )

            vector = resp.data[0].embedding

            # Валидация
            if not vector or len(vector) != self.config.embedding_dim:
                raise ValueError(f"Некорректный размер вектора: {len(vector)} вместо {self.config.embedding_dim}")

            duration = time.time() - start_time
            self.total_tokens += len(text.split())
            self.total_requests += 1

            return vector

        except Exception as e:
            if retry_count < self.config.retry_attempts:
                self.logger.warning(
                    f"⚠️  Ошибка получения embedding, попытка {retry_count + 1}/{self.config.retry_attempts}: {e}")
                time.sleep(self.config.retry_delay * (retry_count + 1))
                return self.get_embedding(text, retry_count + 1)
            else:
                self.logger.error(f"❌ Не удалось получить embedding после {self.config.retry_attempts} попыток: {e}")
                return None

    def prepare_embedding_text(self, record: Dict) -> str:
        """Подготовка текста для embedding"""
        parts = [f"Вопрос: {record['q']}"]

        if record.get('category'):
            parts.append(f"Категория: {record['category']}")

        if record.get('keywords'):
            parts.append(f"Ключевые слова: {record['keywords']}")

        # Добавляем краткий ответ если есть
        if record.get('short_answer'):
            parts.append(f"Краткий ответ: {record['short_answer']}")

        # Добавляем основной ответ
        parts.append(f"Ответ: {record['a']}")

        # Добавляем инструкцию если есть
        if record.get('instructions'):
            parts.append(f"Инструкция: {record['instructions']}")

        # Добавляем документы если есть
        if record.get('documents'):
            parts.append(f"Документы: {record['documents']}")

        # Добавляем законы если есть
        if record.get('law'):
            parts.append(f"Законодательство: {record['law']}")

        return " | ".join(parts)

    def get_statistics(self) -> Dict:
        """Получение статистики использования API"""
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / max(self.total_requests, 1)
        }


# -----------------------------
# РАБОТА С QDRANT
# -----------------------------
class QdrantService:
    """Сервис для работы с Qdrant"""

    def __init__(self, logger: logging.Logger, config: Config):
        self.logger = logger
        self.config = config
        self.client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self._ensure_collection()

    def _ensure_collection(self):
        """Создание коллекции если её нет"""
        if not self.client.collection_exists(self.config.collection_name):
            self.client.create_collection(
                self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"✅ Коллекция создана: {self.config.collection_name}")
        else:
            self.logger.info(f"ℹ️  Коллекция существует: {self.config.collection_name}")

    def get_collection_info(self) -> Dict:
        """Получение информации о коллекции"""
        try:
            info = self.client.get_collection(self.config.collection_name)
            # Совместимость с разными версиями qdrant-client
            points_count = getattr(info, 'points_count', 0)
            vectors_count = getattr(info, 'vectors_count', points_count)  # fallback
            indexed_vectors_count = getattr(info, 'indexed_vectors_count', points_count)
            status = getattr(info, 'status', 'unknown')

            return {
                'points_count': points_count,
                'vectors_count': vectors_count,
                'indexed_vectors_count': indexed_vectors_count,
                'status': status
            }
        except Exception as e:
            self.logger.warning(f"⚠️  Не удалось получить детальную информацию о коллекции: {e}")
            # Пытаемся получить хотя бы количество точек альтернативным способом
            try:
                count_result = self.client.count(collection_name=self.config.collection_name)
                count = count_result.count if hasattr(count_result, 'count') else 0
                return {'points_count': count}
            except:
                return {'points_count': 0}

    def clear_collection(self):
        """Полная очистка коллекции"""
        try:
            self.client.delete_collection(self.config.collection_name)
            self._ensure_collection()
            self.logger.info(f"🗑️  Коллекция очищена: {self.config.collection_name}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка очистки коллекции: {e}")
            raise

    def upsert_points(self, points: List[PointStruct]) -> bool:
        """Загрузка точек в коллекцию"""
        try:
            self.logger.info(f"📤 Загрузка {len(points)} записей в Qdrant...")

            # Батчинг для больших объемов
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=batch
                )

                if len(points) > batch_size:
                    self.logger.info(f"   ✅ Загружено {min(i + batch_size, len(points))}/{len(points)}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки в Qdrant: {e}", exc_info=True)
            return False

    def test_search(self, test_vector: List[float], limit: int = 3) -> List[Dict]:
        """Тестовый поиск"""
        try:
            # Совместимость с разными версиями API
            if hasattr(self.client, 'search'):
                results = self.client.search(
                    collection_name=self.config.collection_name,
                    query_vector=test_vector,
                    limit=limit
                )
            elif hasattr(self.client, 'query_points'):
                # Новый API
                results = self.client.query_points(
                    collection_name=self.config.collection_name,
                    query=test_vector,
                    limit=limit
                ).points
            else:
                # Пробуем через scroll с фильтром
                self.logger.warning("⚠️  Метод поиска не найден, используем альтернативный способ")
                return []

            return [{
                'score': r.score,
                'question': r.payload['metadata']['question'],
                'category': r.payload['metadata'].get('category', 'N/A')
            } for r in results]

        except Exception as e:
            self.logger.warning(f"⚠️  Тестовый поиск недоступен: {e}")
            return []


# -----------------------------
# ОСНОВНОЙ ПРОЦЕССОР
# -----------------------------
class FAQProcessor:
    """Основной процессор для загрузки FAQ"""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.embedding_service = EmbeddingService(logger, config)
        self.qdrant_service = QdrantService(logger, config)
        self.excel_loader = ExcelLoader(logger, config)

    def process_records(self, records: List[Dict]) -> Tuple[List[PointStruct], Dict]:
        """Обработка записей и создание точек для Qdrant"""
        self.logger.info(f"\n🔄 Создание embeddings для {len(records)} записей...")

        points = []
        stats = {
            'successful': 0,
            'failed': 0,
            'failed_items': [],
            'start_time': time.time()
        }

        # Параллельная обработка
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_record, record): record
                for record in records
            }

            for future in as_completed(futures):
                record = futures[future]
                try:
                    point = future.result()
                    if point:
                        points.append(point)
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                        stats['failed_items'].append({
                            'id': record['id'],
                            'question': record['q'][:50]
                        })

                    # Прогресс
                    total_processed = stats['successful'] + stats['failed']
                    if total_processed % 10 == 0:
                        self.logger.info(f"   ✅ Обработано {total_processed}/{len(records)}")

                except Exception as e:
                    stats['failed'] += 1
                    stats['failed_items'].append({
                        'id': record['id'],
                        'question': record['q'][:50],
                        'error': str(e)
                    })
                    self.logger.error(f"❌ Ошибка обработки записи {record['id']}: {e}")

        stats['duration'] = time.time() - stats['start_time']

        return points, stats

    def _process_single_record(self, record: Dict) -> Optional[PointStruct]:
        """Обработка одной записи"""
        try:
            # Подготовка текста для embedding
            embedding_text = self.embedding_service.prepare_embedding_text(record)

            # Получение вектора
            vector = self.embedding_service.get_embedding(embedding_text)
            if not vector:
                return None

            # Создание точки
            point = PointStruct(
                id=record["id"],
                vector=vector,
                payload={
                    "pageContent": f"Вопрос: {record['q']}\nОтвет: {record['a']}",
                    "metadata": {
                        "question": record["q"],
                        "answer": record["a"],
                        "category": record["category"],
                        "keywords": record["keywords"],
                        "source": record["source"],
                        "update_date": record["update_date"],
                        "excel_row": record["row_number"],
                        "source_file": record["source_file"],
                        "load_timestamp": datetime.now().isoformat(),
                        "schema_version": "2.0",
                        "embedding_model": self.config.openai_model
                    }
                }
            )

            return point

        except Exception as e:
            self.logger.error(f"❌ Ошибка создания точки для записи {record.get('id')}: {e}")
            return None

    def run(self, excel_file: str) -> bool:
        """Основной процесс загрузки"""
        try:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"🚀 ЗАПУСК ЗАГРУЗКИ FAQ")
            self.logger.info(f"📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"📁 Файл: {excel_file}")
            self.logger.info(f"🎯 Режим: {self.config.update_mode}")
            self.logger.info(f"{'=' * 60}\n")

            # 1. Загрузка из Excel
            records = self.excel_loader.load(excel_file)
            if not records:
                self.logger.error("❌ Нет данных для загрузки")
                return False

            # 2. Показать категории
            self._show_categories_stats(records)

            # 3. Подтверждение (если нужно)
            if not self.config.auto_confirm and self.config.gui_mode:
                if not self._confirm_upload(len(records), excel_file):
                    self.logger.info("❌ Загрузка отменена пользователем")
                    return False

            # 4. Очистка коллекции (если режим refresh)
            before_count = self.qdrant_service.get_collection_info()['points_count']

            if self.config.update_mode == "refresh":
                self.logger.info(f"🗑️  Режим REFRESH: очистка коллекции...")
                self.qdrant_service.clear_collection()

            # 5. Создание embeddings
            points, processing_stats = self.process_records(records)

            # 6. Загрузка в Qdrant
            if points:
                success = self.qdrant_service.upsert_points(points)
                if not success:
                    return False

                # 7. Проверка результата
                after_count = self.qdrant_service.get_collection_info()['points_count']

                # 8. Тестовый поиск
                if points:
                    self._test_search(records[0], points[0].vector)

                # 9. Итоговая статистика
                self._show_final_stats(
                    processing_stats,
                    before_count,
                    after_count,
                    len(points)
                )

                return True
            else:
                self.logger.error("❌ Не удалось создать ни одного embedding")
                return False

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
            return False

    def _show_categories_stats(self, records: List[Dict]):
        """Показать статистику по категориям"""
        categories = {}
        for record in records:
            cat = record['category'] if record['category'] else "Без категории"
            categories[cat] = categories.get(cat, 0) + 1

        self.logger.info(f"📊 СТАТИСТИКА ПО КАТЕГОРИЯМ:")
        self.logger.info(f"   • Всего категорий: {len(categories)}")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            self.logger.info(f"   • {cat}: {count} записей")
        self.logger.info("")

    def _confirm_upload(self, count: int, excel_file: str) -> bool:
        """Подтверждение загрузки через GUI"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            result = messagebox.askyesno(
                "Подтверждение загрузки",
                f"Загрузить {count} записей в Qdrant?\n\n"
                f"Файл: {os.path.basename(excel_file)}\n"
                f"Коллекция: {self.config.collection_name}\n"
                f"Режим: {self.config.update_mode}"
            )

            root.destroy()
            return result
        except:
            return True

    def _test_search(self, test_record: Dict, test_vector: List[float]):
        """Тестовый поиск"""
        self.logger.info(f"\n🔍 ТЕСТОВЫЙ ПОИСК:")
        self.logger.info(f"   Запрос: '{test_record['q'][:60]}...'")

        results = self.qdrant_service.test_search(test_vector)

        if results:
            for i, result in enumerate(results, 1):
                self.logger.info(f"   {i}. [{result['score']:.3f}] {result['question'][:60]}...")
                if result['category']:
                    self.logger.info(f"      Категория: {result['category']}")
        else:
            self.logger.warning("   ⚠️  Результатов не найдено")

    def _show_final_stats(self, processing_stats: Dict, before_count: int, after_count: int, loaded_count: int):
        """Показать итоговую статистику"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"🎉 ЗАГРУЗКА ЗАВЕРШЕНА!")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"📊 Обработка данных:")
        self.logger.info(f"   • Успешно: {processing_stats['successful']} записей")
        self.logger.info(f"   • Ошибок: {processing_stats['failed']} записей")
        self.logger.info(f"   • Время обработки: {processing_stats['duration']:.2f} сек")

        embedding_stats = self.embedding_service.get_statistics()
        self.logger.info(f"\n📡 API статистика:")
        self.logger.info(f"   • Всего запросов: {embedding_stats['total_requests']}")
        self.logger.info(f"   • Обработано токенов: ~{embedding_stats['total_tokens']}")

        self.logger.info(f"\n💾 Qdrant статистика:")
        self.logger.info(f"   • До загрузки: {before_count} записей")
        self.logger.info(f"   • После загрузки: {after_count} записей")
        self.logger.info(f"   • Загружено: {loaded_count} записей")

        if self.config.update_mode == "refresh":
            self.logger.info(f"   • Режим: ПОЛНАЯ ЗАМЕНА")
        else:
            delta = after_count - before_count
            self.logger.info(f"   • Изменение: {'+' if delta >= 0 else ''}{delta} записей")

        if processing_stats['failed'] > 0:
            self.logger.warning(f"\n⚠️  ОШИБКИ (первые 3):")
            for i, item in enumerate(processing_stats['failed_items'][:3], 1):
                self.logger.warning(f"   {i}. ID {item['id']}: {item['question']}...")
                if 'error' in item:
                    self.logger.warning(f"      Ошибка: {item['error'][:100]}")

        self.logger.info(f"{'=' * 60}\n")

        # GUI уведомление
        if self.config.gui_mode:
            try:
                messagebox.showinfo(
                    "Загрузка завершена",
                    f"✅ Успешно загружено: {processing_stats['successful']} записей\n"
                    f"❌ Ошибок: {processing_stats['failed']} записей\n"
                    f"💾 Всего в коллекции: {after_count} записей\n"
                    f"⏱️  Время: {processing_stats['duration']:.1f} сек"
                )
            except:
                pass


# -----------------------------
# УТИЛИТЫ ДЛЯ GUI
# -----------------------------
def select_excel_file() -> Optional[str]:
    """Диалоговое окно для выбора Excel файла"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Выберите Excel файл с FAQ",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )

        root.destroy()
        return file_path if file_path else None
    except:
        return None


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='FAQ Loader - загрузка базы знаний из Excel в Qdrant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Интерактивный режим (GUI)
  python faq_loader.py

  # Автоматический режим
  python faq_loader.py --file data/faq.xlsx --auto-confirm

  # Полная замена данных
  python faq_loader.py --file data/faq.xlsx --mode refresh --auto-confirm

  # С кастомным конфигом
  python faq_loader.py --config config.yaml --file data/faq.xlsx
        """
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Путь к Excel файлу с FAQ'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Путь к YAML файлу конфигурации'
    )

    parser.add_argument(
        '--mode',
        choices=['upsert', 'refresh', 'incremental'],
        default='upsert',
        help='Режим обновления: upsert (обновление по ID), refresh (полная замена), incremental (только новые)'
    )

    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='Автоматическое подтверждение без GUI диалогов'
    )

    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Отключить GUI элементы (для автоматизации)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Имя файла для логов'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Размер батча для обработки (по умолчанию: 20)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Количество параллельных workers (по умолчанию: 5)'
    )

    parser.add_argument(
        '--max-answer-length',
        type=int,
        default=5000,
        help='Максимальная длина ответа в символах (по умолчанию: 5000)'
    )

    parser.add_argument(
        '--max-question-length',
        type=int,
        default=1000,
        help='Максимальная длина вопроса в символах (по умолчанию: 1000)'
    )

    return parser.parse_args()


# -----------------------------
# MAIN
# -----------------------------
def main():
    """Главная функция"""
    # Парсинг аргументов
    args = parse_args()

    # Настройка логирования
    logger = setup_logging(args.log_file)

    try:
        # Загрузка конфигурации
        if args.config:
            config = Config.from_yaml(args.config)
        else:
            config = Config.from_env()

        # Применение CLI аргументов
        config.update_mode = args.mode
        config.auto_confirm = args.auto_confirm
        config.gui_mode = not args.no_gui

        if args.batch_size:
            config.batch_size = args.batch_size
        if args.workers:
            config.max_workers = args.workers
        if args.max_answer_length:
            config.max_answer_length = args.max_answer_length
        if args.max_question_length:
            config.max_question_length = args.max_question_length

        # Выбор файла
        if args.file:
            excel_file = args.file
            logger.info(f"📁 Файл из аргументов: {excel_file}")
        else:
            if config.gui_mode:
                logger.info("📂 Выберите Excel файл...")
                excel_file = select_excel_file()
            else:
                logger.error("❌ В режиме --no-gui необходимо указать --file")
                return 1

        if not excel_file:
            logger.error("❌ Файл не выбран")
            return 1

        if not os.path.exists(excel_file):
            logger.error(f"❌ Файл не найден: {excel_file}")
            return 1

        # Создание процессора и запуск
        processor = FAQProcessor(config, logger)
        success = processor.run(excel_file)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Программа прервана пользователем (Ctrl+C)")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Критическая ошибка: {e}", exc_info=True)
        if args.no_gui:
            return 1
        else:
            try:
                messagebox.showerror("Критическая ошибка", f"Произошла ошибка:\n\n{str(e)}")
            except:
                pass
            return 1


if __name__ == "__main__":
    sys.exit(main())