from transformers import AutoTokenizer

if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    one_token_nums = []

    for i in range(0, 1001):
        res = tokenizer("{}".format(i), add_special_tokens=False)
        # print(len(res['input_ids']))
        if len(res['input_ids']) != 1:
            one_token_nums.append(i)
    
    # write to file
    # with open("one_token_nums.txt", "w") as f:
    #     f.write(str(one_token_nums))

    print(one_token_nums)
    # print(tokenizer.sep_token)
