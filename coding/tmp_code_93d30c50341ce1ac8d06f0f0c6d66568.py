from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

# Load the pre-trained model and processor
model = LlavaForConditionalGeneration.from_pretrained("ahmed-masry/ChartInstruct-LLama2", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/ChartInstruct-LLama2")

# Prepare the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the image
image_path = "D:\\Dai_hoc\\Cac_mon_hoc\\Ky_5\\Nhap_mon_du_lieu_lon\\CuoiKy\\MyAgent\\Chart\\chart_example_1.png"
image = Image.open(image_path).convert("RGB")

# Define the question
input_text = "What is the share of responders who prefer Facebook Messenger in the 18-29 age group?"

# Prepare the input prompt
input_prompt = f"<image>\nQuestion: {input_text} Answer: "

# Process the input
inputs = processor(text=input_prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
prompt_length = inputs['input_ids'].shape[1]

# Generate the answer using beam search
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)

# Decode the output
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Print the result
print(output_text)
