from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import random

random.seed(42)
MODEL_PATH = './logs/1l1h_no_pairs_rep/trained'
DATA_PATH = './data/no_pairs_rep_test.csv' 
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# load trained model in logs with pipeline
pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

data_eval = pd.read_csv(DATA_PATH, dtype=str)


# for i in range(1, 4):
#     for j in range(1, 4):
#         result = pipe(f'{i} {j} [SEP] [MASK]')
#         print(f"Input: {i} {j} [SEP] [MASK]")
#         for prediction in result:
#             word = prediction['token_str']
#             print(f"Predicted word: {word} with score {prediction['score']:.4f}")
#         print()



# for i in range(4):
#     rnd = random.randint(0, 341)
#     result = pipe(f'{rnd} {rnd} [SEP] [MASK]')
#     print(f"Input: {rnd} {rnd} [SEP] [MASK]")
#     for prediction in result:
#         word = prediction['token_str']
#         print(f"Predicted word: {word} with score {prediction['score']:.4f}")
#     print()

num1 = [125, 27, 19, 118]
num2 = [126, 26, 12, 12]
for i in range(len(num1)):
    result = pipe(f'{num1[i]} {num2[i]} [SEP] [MASK]')
    print(f"Input: {num1[i]} {num2[i]} [SEP] [MASK]")
    for prediction in result:
        word = prediction['token_str']
        print(f"Predicted word: {word} with score {prediction['score']:.4f}")
    print()

# text = list(data_eval['text'])
# # shuffle the list text
# random.shuffle(text)
# for i in range(10):
#     result = pipe(text[i])
#     print(f"Input: {text[i]}")
#     for prediction in result:
#         word = prediction['token_str']
#         print(f"Predicted word: {word} with score {prediction['score']:.4f}")
#     print()