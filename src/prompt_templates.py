import json

from llm_client import ask_gigachat


def classify_news_prompt(news_text: str) -> str:
    """Промпт для классификации русскоязычной новостной статьи по 11 темам.

    Соответствует датасету data-silence/rus_news_classifier (data/processed/clean.csv),
    где колонка topic принимает одно из значений:
    climate, conflicts, culture, economy, gloss, health, politics,
    science, society, sports, travel.
    """
    prompt = f"""Ты — редактор новостной агрегаторной платформы. Твоя задача — определить тематику новости и вернуть ТОЛЬКО валидный JSON без markdown-обёртки (без ```json ... ```), без пояснений и лишнего текста.

Новость для анализа:
\"\"\"{news_text}\"\"\"

Инструкция:
1. Выбери ОДНУ тему (topic) строго из этого списка, любые другие значения недопустимы:
   - "climate" — экология, погода, климат, природные катаклизмы, выбросы CO2
   - "conflicts" — военные действия, теракты, нападения, столкновения
   - "culture" — кино, музыка, театр, искусство, знаменитости, шоу-бизнес
   - "economy" — финансы, бизнес, рынки, валюта, цены, компании, банки
   - "gloss" — мода, стиль, beauty, светская хроника
   - "health" — медицина, болезни, врачи, ЗОЖ, питание, фитнес
   - "politics" — деятельность властей, законы, выборы, дипломатия, партии
   - "science" — наука, технологии, IT, космос, исследования, изобретения
   - "society" — криминал, происшествия, бытовые истории, люди, отношения
   - "sports" — спорт, соревнования, клубы, атлеты
   - "travel" — туризм, путешествия, авиакомпании, курорты, отели
2. Напиши краткое резюме новости в 1-2 предложениях (summary).
3. Выпиши 2-4 ключевых сущности из текста: имена, организации, топонимы (entities).
4. Оцени тональность новости (sentiment): "позитивная", "нейтральная" или "негативная".

Формат ответа — строго следующий JSON:
{{
  "topic": "...",
  "summary": "...",
  "entities": ["...", "..."],
  "sentiment": "..."
}}

Пример правильного ответа на новость "Основатель Alibaba Джек Ма разбогател на 1,4 миллиарда долларов после появления на публике. Рыночная стоимость увеличилась на 58 миллиардов долларов":
{{
  "topic": "economy",
  "summary": "Состояние Джека Ма выросло на 1,4 млрд долларов после его первого за три месяца публичного появления. Капитализация Alibaba выросла на 58 млрд долларов.",
  "entities": ["Джек Ма", "Alibaba", "Forbes"],
  "sentiment": "позитивная"
}}

Верни только JSON.
    """
    return prompt


if __name__ == "__main__":
    # Берём реальную новость из нашего датасета (колонки text, topic)
    import pandas as pd
    from pathlib import Path

    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "clean.csv"
    df = pd.read_csv(DATA_PATH)
    row = df.iloc[0]
    news_text = row["text"]
    true_topic = row["topic"]

    print("1. Формируем промпт под классификацию новости...")
    final_prompt = classify_news_prompt(news_text)

    print("2. Отправляем запрос в GigaChat...")
    raw_response = ask_gigachat(final_prompt, temperature=0.1)

    print(f"\nСырой ответ от модели:\n{raw_response}\n")

    print("3. Проверяем валидность полученного JSON...")
    try:
        parsed_json = json.loads(raw_response.strip())
        print("Успех! Данные успешно преобразованы в Python dict:")
        print(f"Тема (topic): {parsed_json.get('topic')}")
        print(f"Резюме (summary): {parsed_json.get('summary')}")
        print(f"Сущности (entities): {parsed_json.get('entities')}")
        print(f"Тональность (sentiment): {parsed_json.get('sentiment')}")

        if parsed_json.get("topic") == true_topic:
            print(f"\n✅ Предсказание совпало с разметкой датасета: {true_topic}")
        else:
            print(f"\n❌ Расхождение: модель сказала '{{}}', в датасете '{{}}'".format(parsed_json.get("topic"), true_topic))
    except json.JSONDecodeError:
        print("Ошибка: Модель нарушила формат и вернула невалидный JSON.")