import os
import sys
import json
import re
import subprocess
import ollama

# Конфигурация
MODEL_NAME = "qwen2.5-coder:7b" 
OUTPUT_DIR = "output"

def get_llm_response(prompt, system_prompt=""):
    """Отправка запроса в Ollama"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    print(f"[*] Генерация ответа от модели {MODEL_NAME}...")
    response = ollama.chat(model=MODEL_NAME, messages=messages, options={"temperature": 0.3})
    return response['message']['content']

def extract_code_block(text, language="python"):
    """Извлечение кода из markdown-оберток ```python ... ```"""
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Если модель не обернула в блок, возвращаем как есть
    return text.strip()

def extract_json(text):
    """Извлечение валидного JSON из текста"""
    # Ищем первый { или [ и последний } или ]
    start = min(text.find('{'), text.find('['))
    if start == -1: return text
    end = max(text.rfind('}'), text.rfind(']')) + 1
    return text[start:end].strip()

def generate_test_cases(description):
    system = "Ты — опытный QA Engineer. Отвечай строго на русском языке."
    prompt = f"""
    На основе описания функционала сгенерируй список тест-кейсов в формате Markdown.
    Формат каждого кейса:
    ### TC-001: [Название]
    - **Предусловия:** ...
    - **Шаги:** 1. ... 2. ...
    - **Ожидаемый результат:** ...
    
    Описание функционала:
    {description}
    """
    return get_llm_response(prompt, system)

def generate_test_data(description):
    system = "Ты — QA Engineer. Генерируй строго валидный JSON. Никакого текста до и после JSON."
    prompt = f"""
    Сгенерируй массив тестовых данных в формате JSON для параметризации pytest.
    Каждый объект должен содержать поля: "username", "password", "confirm_password", "expected_status" (строка: "success" или "error: [причина]").
    Покрой позитивные и негативные сценарии (пустые поля, несовпадение паролей, короткие пароли).
    
    Описание функционала:
    {description}
    """
    raw_json = get_llm_response(prompt, system)
    clean_json = extract_json(raw_json)
    # Проверяем валидность
    try:
        json.loads(clean_json)
        return clean_json
    except json.JSONDecodeError:
        print("[!] Модель сгенерировала невалидный JSON, пробуем исправить...")
        return clean_json # Сохраним как есть, пусть pytest упадет, это часть реальности работы с LLM

def generate_pytest_code(description):
    system = "Ты — SDET (Python). Пиши чистый код. Возвращай ТОЛЬКО код на Python, без markdown-оберток и пояснений."
    prompt = f"""
    Напиши автотест на pytest для проверки бизнес-логики регистрации.
    ВАЖНО: Так как у нас нет UI и бэкенда, напиши мок-функцию `mock_register(username, password, confirm_password)`, которая проверяет логику:
    1. Имя не пустое.
    2. Пароль минимум 8 символов.
    3. Пароль и подтверждение совпадают.
    Функция должна возвращать "success" или строку с ошибкой.
    
    Тест должен читать данные из файла `test_data.json` (используй `os.path.dirname(__file__)` для пути) и параметризовать тесты через `@pytest.mark.parametrize`.
    
    Описание функционала:
    {description}
    """
    raw_code = get_llm_response(prompt, system)
    return extract_code_block(raw_code, "python")

def main():
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
    else:
        description = """
        Форма регистрации пользователя на сайте otus.ru. 
        Форма имеет 3 поля ввода: имя пользователя, пароль и подтверждение пароля. 
        Также есть кнопка «Зарегистрироваться».
        """
        print(f"[*] Используется описание по умолчанию:\n{description}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Тест-кейсы
    print("[1/3] Генерация тест-кейсов...")
    cases_md = generate_test_cases(description)
    with open(os.path.join(OUTPUT_DIR, "test_cases.md"), "w", encoding="utf-8") as f:
        f.write(cases_md)

    # 2. Тестовые данные
    print("[2/3] Генерация тестовых данных...")
    data_json = generate_test_data(description)
    with open(os.path.join(OUTPUT_DIR, "test_data.json"), "w", encoding="utf-8") as f:
        f.write(data_json)

    # 3. Автотесты
    print("[3/3] Генерация кода автотестов...")
    pytest_code = generate_pytest_code(description)
    pytest_file = os.path.join(OUTPUT_DIR, "test_registration.py")
    with open(pytest_file, "w", encoding="utf-8") as f:
        f.write(pytest_code)

    print("\n[✓] Артефакты успешно сохранены в папку 'output/'.")
    
    # 4. Запуск pytest
    print("\n[*] Запуск pytest...")
    result = subprocess.run(["pytest", pytest_file, "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[!] Ошибки при выполнении тестов:")
        print(result.stderr)

if __name__ == "__main__":
    main()