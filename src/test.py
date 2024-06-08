from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import random
from tqdm import tqdm

MODEL_PATH = './logs/1l1h_no_same_permutation/trained'
DATA_PATH = './data/val_data_2.csv' 

def test_specific_samples(list1, list2):
    """
    Test on samples in list1 and list2, print all the predictions. 
    list1 and list2 should be of the same length.

    e.g.
        list1 = [125, 27, 19, 118]
        list2 = [126, 26, 12, 12]
    """
    for i in range(len(list1)):
        result = pipe(f'{list1[i]} {list2[i]} [SEP] [MASK]')
        print(f"Input: {list1[i]} {list2[i]} [SEP] [MASK]")
        for prediction in result:
            word = prediction['token_str']
            print(f"Predicted word: {word} with score {prediction['score']:.4f}")
        print()



def test_random_samples(data_eval, num_samples=10):
    """
    Test on random samples in the test set, print all the predictions
    """
    text = list(data_eval['text'])
    random.shuffle(text)
    for i in range(num_samples):
        result = pipe(text[i])
        print(f"Input: {text[i]}")
        for prediction in result:
            word = prediction['token_str']
            print(f"Predicted word: {word} with score {prediction['score']:.4f}")
        print()


def test_whole_set_top1(data_eval, output_file=None):
    """
    Test on the whole test set and print the wrong predictions for the top 1 prediction. 
    """
    text = list(data_eval['text'])
    labels = list(data_eval['labels'])
    # if shuffle:
    #     c = list(zip(text, labels))
    #     random.shuffle(c)
    #     text, labels = zip(*c)
    #     text = list(text)
    #     labels = list(labels)
    print(f"Number of samples in the test set: {len(text)}")

    wrong_predictions = []
    for i in tqdm(range(len(text)), desc="Testing on the whole test set"):
        result = pipe(text[i])
        top1_prediction = result[0]
        word = top1_prediction['token_str'] # get top1 prediction
        score = top1_prediction['score']
        if word != labels[i]:
            wrong_predictions.append((text[i], word, labels[i], score))

    print()

    if output_file:
        with open(output_file, 'w') as f:
            for i in range(len(wrong_predictions)):
                f.write(f"Input: {wrong_predictions[i][0]}\n")
                f.write(f"Predicted word: {wrong_predictions[i][1]}\n")
                f.write(f"True word: {wrong_predictions[i][2]}\n")
                f.write(f"Score: {wrong_predictions[i][3]:.4f}\n\n")
        print(f"Number of samples in the test set: {len(text)}")
        print(f"Number of wrong predictions: {len(wrong_predictions)}")
        print(f"Accuracy: {(len(text) - len(wrong_predictions)) / len(text) * 100:.2f}%")
        print(f"Wrong predictions are saved to {output_file}")
    else:
        print("Wrong predictions:")
        for i in range(len(wrong_predictions)):
            print(f"Input: {wrong_predictions[i][0]}")
            print(f"Predicted word: {wrong_predictions[i][1]}")
            print(f"True word: {wrong_predictions[i][2]}")
            print(f"Score: {wrong_predictions[i][3]:.4f}")
            print()
        print(f"Number of samples in the test set: {len(text)}")
        print(f"Number of wrong predictions: {len(wrong_predictions)}")
        print(f"Accuracy: {(len(text) - len(wrong_predictions)) / len(text) * 100:.2f}%")


if __name__ == '__main__':
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Number of trainable parameters: {model.num_parameters()}")
    print()
    # print(model)

    # load trained model in logs with pipeline
    pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    data_eval = pd.read_csv(DATA_PATH, dtype=str)

    test_whole_set_top1(data_eval, output_file='./logs/1l1h_no_same_permutation/wrong_predictions.txt')