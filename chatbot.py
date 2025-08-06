from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

while True:
# Get the input data from the user
    input_text = input("> ")

    # Exit if user types quit/exit
    if input_text.lower() in ["quit", "exit"]:
        break

#Tokenize the input text and history 
    full_input = " </s ".join(conversation_history + [input_text])
    inputs = tokenizer(full_input, return_tensors="pt")
#Generate the response from the model
    outputs = model.generate(**inputs)

    #Decode the response 
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print("Derrick:", response)

    #Add Interaction to conversaton history
    conversation_history.append(input_text)
    conversation_history.append(response)

