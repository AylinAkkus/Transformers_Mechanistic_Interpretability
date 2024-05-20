from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer

model_path = './logs/simple_model/checkpoint-20000'
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(model)

# load trained model in logs with pipeline
pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

result = pipe('1 2 [SEP] [MASK]')
# print(result)

for prediction in result:
    word = prediction['token_str']
    print(f"Predicted word: {word} with score {prediction['score']:.4f}")