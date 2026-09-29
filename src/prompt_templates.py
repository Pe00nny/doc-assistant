import json
from llm_client import ask_gigachat

def analyze_review_prompt(review_text: str) -> str:
    prompt = f"""
ДОМА НАПИСАТ ПРОМПТ
    """
    return prompt

if __name__ == "__main__":
    # Сырой отзыв для анализа
    user_review = """ Купил эти наушники вчера. Звук чистый, объемный, за свои деньги топ. Однако амбюшуры слишком жесткие, уши начинают болеть через час использования"""
    
    print("1. Формируем сложный промпт...")
    final_prompt = analyze_review_prompt(user_review)
    
    print("2. Отправляем запрос в GigaChat...")
    raw_response = ask_gigachat(final_prompt, temperature=0.1)
    
    print(f"\nСырой ответ от модели:\n{raw_response}\n")
    
    print("3. Проверяем валидность полученного JSON...")
    try:
        parsed_json = json.loads(raw_response.strip())
        print("Успех! Данные успешно преобразованы в Python dict: ")
        print(f"Тональность: {parsed_json.get('sentiment')}")
        print(f"Плюсы: {parsed_json.get('pros')}")
        print(f"Минсы: {parsed_json.get('cons')}")
    except json.JSONDecodeError:
        print("Ошибка: Модель нарушила формат и вернула невалидный JSON.")
        
    