from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

model_name = "khhuang/chartve"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

image_path = "StockPriceYTDNVDA.png"

def format_query(sentence):
    return f"Does the image entails this statement: \"{sentence}\"?"

# Format text inputs
CAPTION_SENTENCE = "NVDA and TESLA stock price change YTD."
query = format_query(CAPTION_SENTENCE)

# Encode chart figure and tokenize text
img = Image.open(image_path)
pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
pixel_values = pixel_values
decoder_input_ids = processor.tokenizer(query, add_special_tokens=False, return_tensors="pt", max_length=510).input_ids


outputs = model(pixel_values, decoder_input_ids=decoder_input_ids)
# print(outputs.logits)

# positive_logit = outputs['logits'].squeeze()[-1,49922]
# negative_logit = outputs['logits'].squeeze()[-1,2334] 

# Probe the probability of generating "yes"
binary_entail_prob_positive = torch.nn.functional.softmax(outputs['logits'].squeeze()[-1,[2334, 49922]], dim=0)[1].item()

logits = outputs.logits.squeeze()  # Bỏ chiều batch
positive_logit = logits[-1, 49922]  # Logit của token "yes"
negative_logit = logits[-1, 2334]   # Logit của token "no"

# Tính xác suất của "yes"
probabilities = torch.nn.functional.softmax(logits[-1, [2334, 49922]], dim=0)
binary_entail_prob_positive = probabilities[1].item()  # Xác suất "yes"
print(binary_entail_prob_positive)


# binary_entail_prob_positive corresponds to the computed probability that the chart entails the caption sentence.
