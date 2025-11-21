from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "/home/kexin/hd1/zkf/Qwen3-VL", 
    dtype="auto", device_map="auto", 
    # attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained("/home/kexin/hd1/zkf/Qwen3-VL")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/kexin/hd1/zkf/IndOVSS/imgs/rag_test1.jpeg",
            },
            {"type": "text", 
             "text": 
                """
                You are an image analysis expert.

                ## Prompt:
                Headphone consist of two round earcups and an arched headband.

                ## Task: 
                Describe the structure of the pantograph slider. Your answer should be formatted as shown in the ## Prompt. 

                ## Requirements: 
                Only structural information is required.
                """
            },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)