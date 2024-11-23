# bot.py

import asyncio 
from aiogram import Bot, Dispatcher
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile, InputMediaPhoto, InputFile
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import logging
from datetime import datetime
from glob import glob
import pandas as pd
from newutils import MatchPredictor
import requests
from llmanalys import csv_to_string, api_request

def get_sport_name(team_choice):
    if team_choice == "Сборная по Хоккею":
        return "hockey"
    elif team_choice == "Сборная по Футболу":
        return "football"
    else:
        print('Ошибка определения вида спорта')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_TOKEN = "7592943669:AAEtdkKW1wQcXK2vwrRbO5SpahciPHBJlnQ" 

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

helper = MatchPredictor()
for i in ['hockey', 'football']:
    helper.train_model(i)
    helper.save_all_plots(i, save=True)

# Словарь для хранения выборов пользователей
user_choices = {}

class Form(StatesGroup):
    waiting_for_match_result = State()
    waiting_for_match_prediction = State()


# Клавиатура для выбора сборной
def main_menu():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Сборная по Хоккею"),
                KeyboardButton(text="Сборная по Футболу"),
            ]
        ],
        resize_keyboard=True
    )
    return keyboard

# Клавиатура после выбора сборной
def secondary_menu():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Просмотреть последние 5 матчей")],
            [KeyboardButton(text="Анализ по графикам")],
            [KeyboardButton(text="Добавить матч")],
            [KeyboardButton(text="Прогноз результата матча")],
            [KeyboardButton(text="Анализ LLM")],  # Добавлена пятая кнопка
        ],
        resize_keyboard=True
    )
    return keyboard

async def get_user_team(message: Message):
    user_id = message.from_user.id
    team = user_choices.get(user_id)
    if not team:
        await message.answer("Пожалуйста, сначала выберите сборную.", reply_markup=main_menu())
    return team

@dp.message(Command(commands=["start"]))
async def start_command(message: Message):
    user_choices.pop(message.from_user.id, None)
    await message.answer(
        "Добрый день! Я - бот аналитики для 'Медведей Политеха'.\nВыберите сборную:",
        reply_markup=main_menu()
    )

@dp.message(lambda message: message.text in ["Сборная по Хоккею", "Сборная по Футболу"])
async def choose_team(message: Message):
    user_id = message.from_user.id
    choice = message.text
    user_choices[user_id] = choice
    photo_mapping = {
        "Сборная по Хоккею": "hockey.jpg",
        "Сборная по Футболу": "futbol.jpg"
    }

    # Get the corresponding photo filename
    photo_filename = photo_mapping.get(choice)
    photo = FSInputFile(photo_filename)
    await message.answer_photo(
        photo=photo,
        caption="Отлично! Что хотите узнать?",
        reply_markup=secondary_menu()
    )

@dp.message(lambda message: message.text == "Просмотреть последние 5 матчей")
async def view_last_matches(message: Message):
    team = await get_user_team(message)
    if not team:
        return
    sport_name = get_sport_name(team)
    df = pd.read_csv(f"{sport_name}.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date'], ascending=False)[:5]
    ans = f"Последние матчи сборной по {team.split()[-1]}:\n"
    for ind, i in enumerate(range(len(df))):
        row = df.iloc[i]
        try:
            ans += f"{ind+1}) {row['date'].strftime('%d.%m.%Y')} {row['time']} {row['team2']} {row['score1']}:{row['score2']}\n"
        except:
            ans += f"{ind+1}) {str(row['date']).split(' ')[0]} {row['time']} {row['team2']} {row['score1']}:{row['score2']}\n"
    await message.answer(ans)

@dp.message(lambda message: message.text == "Анализ по графикам")
async def analyze_team_games(message: Message):
    team = await get_user_team(message)
    if not team:
        return
    sport_name = get_sport_name(team)
    
    if not sport_name:
        await message.answer("Произошла ошибка с определением вида спорта.")
        return

    tosend = []
    for photo_path in glob(f'plots/{sport_name}/*'):
        try:
            photo = InputMediaPhoto(media=FSInputFile(path=photo_path))
            tosend.append(photo)
        except Exception as e:
            logger.error(f"Ошибка при отправке изображения {photo_path}: {e}")
            await message.answer(f"Не удалось отправить изображение {photo_path}")
    if tosend:
        await message.answer_media_group(media=tosend)
    else:
        await message.answer("Нет доступных графиков для отображения.")

@dp.message(lambda message: message.text == "Добавить матч")
async def add_match(message: Message, state: FSMContext):
    team = await get_user_team(message)
    if not team:
        return
    await message.answer(f"Введите результаты матча в формате:\nДата Время Команда Счет\nНапример: 27.09.2024 17:00 Зенит 4:0")
    await state.set_state(Form.waiting_for_match_result)

@dp.message(lambda message: message.text == "Прогноз результата матча")
async def predict_match_handler(message: Message, state: FSMContext):
    team = await get_user_team(message)
    if not team:
        return
    await message.answer(f"Введите дату, время и команду-противника предстоящего матча в формате:\nДата Время Команда\nНапример: 27.09.2024 17:00 Зенит")
    await state.set_state(Form.waiting_for_match_prediction)

@dp.message(lambda message: message.text == "Анализ LLM")  # Обработчик для новой кнопки
async def llm_analysis(message: Message):
    team = await get_user_team(message)
    if not team:
        return
    sport_name = get_sport_name(team)
    if not sport_name:
        await message.answer("Произошла ошибка с определением вида спорта.")
        return

    # Вызов функций csv_to_string и api_request, вывод результата пользователю
    csv_string = csv_to_string(sport_name)
    await message.answer(str(api_request(csv_string)['result']))

@dp.message(Form.waiting_for_match_result)
async def handle_match_result(message: Message, state: FSMContext):
    match_result = message.text.strip()
    user_id = message.from_user.id
    team = user_choices.get(user_id)
    
    if not team:
        await message.answer("Произошла ошибка. Пожалуйста, начните сначала.", reply_markup=main_menu())
        await state.clear()
        return

    parts = match_result.split(' ')
    if len(parts) != 4:
        await message.answer("Неверный формат сообщения.\nПожалуйста, введите данные в формате:\nДата Время Команда Счет\nНапример: 27.09.2024 17:00 Зенит 4:0")
        return

    date_str, time_str, opponent_team, score = parts

    # Проверка даты и времени
    try:
        match_datetime = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
    except ValueError:
        await message.answer("Неверный формат даты или времени.\nПожалуйста, используйте формат ДД.ММ.ГГГГ ЧЧ:ММ")
        return

    # Проверка счета
    if ':' not in score:
        await message.answer("Неверный формат счета.\nПожалуйста, используйте формат Число:Число (например, 4:0)")
        return

    try:
        score_home, score_away = map(int, score.split(':'))
    except ValueError:
        await message.answer("Счет должен содержать два числа, разделенных двоеточием.")
        return

    # Сохраняем данные
    nuzhno = f"{match_datetime.strftime('%d.%m.%Y')} {time_str} {opponent_team} {score_home}:{score_away}"
    sport_name = get_sport_name(team)
    helper.remake_data(sport_name, nuzhno)
    helper.train_model(sport_name)
    helper.save_all_plots(sport_name, save=True)
    await message.answer("Результаты матча сохранены и модели обновлены.")
    await state.clear()

@dp.message(Form.waiting_for_match_prediction)
async def handle_match_prediction(message: Message, state: FSMContext):
    prediction = message.text.strip()
    user_id = message.from_user.id
    team = user_choices.get(user_id)

    if not team:
        await message.answer("Произошла ошибка. Пожалуйста, начните сначала.", reply_markup=main_menu())
        await state.clear()
        return

    parts = prediction.split(' ')
    if len(parts) != 3:
        await message.answer("Неверный формат сообщения.\nПожалуйста, введите данные в формате:\nДата Время Команда\nНапример: 27.09.2024 17:00 Зенит")
        return

    date_str, time_str, opponent_team = parts

    # Проверка даты и времени
    try:
        match_datetime = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
    except ValueError:
        await message.answer("Неверный формат даты или времени.\nПожалуйста, используйте формат ДД.ММ.ГГГГ ЧЧ:ММ")
        return

    # Сохраняем данные
    nuzhno = f"{match_datetime.strftime('%d.%m.%Y')} {time_str} {opponent_team}"
    sport_name = get_sport_name(team)
    preds = helper.predict_match(sport_name, nuzhno)
    await message.answer(
        text=f"Матч прогнозируется с разницей в {round(preds['score_diff_pred'], 2)} очка.\n"
             f"Вероятность победы: {round(preds['win_probability'] * 100, 2)}%"
    )
    await state.clear()

@dp.message()
async def default_handler(message: Message):
    team = user_choices.get(message.from_user.id)
    if team:
        await message.answer(
            "Пожалуйста, используйте кнопки для взаимодействия с ботом.",
            reply_markup=secondary_menu()
        )
    else:
        await message.answer(
            "Пожалуйста, используйте кнопки для взаимодействия с ботом.",
            reply_markup=main_menu()
        )

async def main():
    # Регистрация всех обработчиков
    dp.message.register(start_command, Command(commands=["start"]))
    dp.message.register(choose_team, lambda message: message.text in ["Сборная по Хоккею", "Сборная по Футболу"])
    dp.message.register(view_last_matches, lambda message: message.text == "Просмотреть последние 5 матчей")
    dp.message.register(analyze_team_games, lambda message: message.text == "Анализ по графикам")
    dp.message.register(add_match, lambda message: message.text == "Добавить матч")
    dp.message.register(predict_match_handler, lambda message: message.text == "Прогноз результата матча")
    dp.message.register(llm_analysis, lambda message: message.text == "Анализ LLM")  # Регистрация нового обработчика
    dp.message.register(handle_match_result, Form.waiting_for_match_result)
    dp.message.register(handle_match_prediction, Form.waiting_for_match_prediction)
    dp.message.register(default_handler)
    
    # Запуск бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
