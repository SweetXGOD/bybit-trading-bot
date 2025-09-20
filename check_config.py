import os
from dotenv import load_dotenv

load_dotenv()

print('Проверка конфигурации:')
print('TELEGRAM_BOT_TOKEN:', 'установлен' if os.getenv('TELEGRAM_BOT_TOKEN') else 'ОШИБКА')
print('TELEGRAM_CHAT_ID:', 'установлен' if os.getenv('TELEGRAM_CHAT_ID') else 'ОШИБКА')
print('HUOBI_API_KEY:', 'установлен' if os.getenv('HUOBI_API_KEY') else 'ОШИБКА')
print('HUOBI_SECRET_KEY:', 'установлен' if os.getenv('HUOBI_SECRET_KEY') else 'ОШИБКА')
print('TARGET_EQUITY:', os.getenv('TARGET_EQUITY'))