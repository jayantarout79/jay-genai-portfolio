def clean_user_input(user_prompt):
    text= user_prompt.strip().replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split()) 
    return text

def add_context(cleaned_prompt, additional_info):
    if additional_info:
        return f" {additional_info} {cleaned_prompt}"
    return cleaned_prompt

if __name__ == "__main__":
    raw_input = "   What are the key metrics   \n for churn?   "
    cleaned_prompt = clean_user_input(raw_input)
    additional_info = "Act as a helpful assistant."
    prompt = add_context(cleaned_prompt, additional_info)
    print(prompt)