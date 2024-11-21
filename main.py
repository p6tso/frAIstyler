import numpy as np
import logging
import asyncio
import sys
from aiogram import Bot, Dispatcher, types
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.command import Command
from secret import BOT_TG_TOKEN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
bot = Bot(token=BOT_TG_TOKEN)
dp = Dispatcher()

user2model = {}


def generate_text_with_temperature(model, tokenizer, seed_text, next_words, max_sequence_len, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]

        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        predicted = np.random.choice(len(predictions), p=predictions)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


model_data = {'AQUAKEY': {'model': 'models_lstm/aquakey_lstm.keras',
                          'text': 'text/AQUAKEY_edited.txt'},
              'Ежемесячные': {'model': 'models_lstm/kpss_lstm.keras',
                              'text': 'text/Ежемесячные_edited.txt'},
              'KUNTEYNIR': {'model': 'models_lstm/kunteynir_lstm.keras',
                            'text': 'text/KUNTEYNIR_edited.txt'}}

models = {}


def use_model(data: str = 'AQUAKEY'):
    with open(f'{model_data[data]["text"]}', 'r') as file:
        text = file.read()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    model = load_model(f'{model_data[data]["model"]}')
    models[data] = (model, tokenizer)


@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.answer("Добро пожаловать! Используйте команду /model для выбора модели.")


@dp.message(Command("model"))
async def cmd_model(message: types.Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="AQUAKEY",
        callback_data="AQUAKEY")
    )
    builder.add(types.InlineKeyboardButton(
        text="KUNTEYNIR",
        callback_data="KUNTEYNIR")
    )
    builder.add(types.InlineKeyboardButton(
        text="Ежемесячные",
        callback_data="Ежемесячные")
    )
    await message.answer(
        "Нажмите на кнопку, чтобы выбрать модель",
        reply_markup=builder.as_markup()
    )


@dp.callback_query(lambda callback: callback.data in ["AQUAKEY", "KUNTEYNIR", "Ежемесячные"])
async def set_user_model(callback: types.CallbackQuery):
    user = callback.from_user.id
    chosen_model = callback.data
    user2model[user] = chosen_model
    await callback.message.answer(f"Текущая модель: {chosen_model}")
    await callback.answer()


@dp.message(lambda message: message.text)
async def generate(message: types.Message):
    user = message.from_user.id
    chosen_model = user2model.get(user, 'Ежемесячные')
    model, tokenizer = models[chosen_model]
    seed_text = message.text
    generated_text = generate_text_with_temperature(model, tokenizer, seed_text, 30, 50, 0.7)
    await message.answer(generated_text)


async def main():
    for model in model_data.keys():
        use_model(model)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
