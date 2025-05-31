from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd

model_name = "khhuang/chart-to-table"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

image_path = "chart_example_1.png"

# Format text inputs

input_prompt = "<data_table_generation> <s_answer>"

# Encode chart figure and tokenize text
img = Image.open(image_path)
pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
pixel_values = pixel_values
decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt", max_length=510).input_ids#.squeeze(0)


outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
print(sequence)
extracted_table = sequence.split("<s_answer>")[1].strip()
print(extracted_table)

cleaned_data = extracted_table.split("&&&")
data_pairs = []

for item in cleaned_data:
    parts = item.split("|")
    if len(parts) == 2:  # Lấy cặp ngày và giá
        date = parts[0].strip()
        price = parts[1].strip()
        data_pairs.append((date, price))

# Chuyển thành DataFrame
df = pd.DataFrame(data_pairs, columns=["Date", "Adjusted Closing Price"])

# Xóa các mục không hợp lệ hoặc bị lặp
df = df[df["Date"].notna()]  # Xóa mục có "None"
df = df[df["Adjusted Closing Price"].notna()]
df = df.drop_duplicates()

print(df)