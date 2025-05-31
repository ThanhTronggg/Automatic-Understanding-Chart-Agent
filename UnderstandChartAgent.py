from autogen import (
    Agent,
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    config_list_from_json,
)

config_list = config_list_from_json(env_or_file=r"MyAgent\OAI_CONFIG_LIST")

boss = UserProxyAgent(
    name="Boss",
    # human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

qa_agent = AssistantAgent(
    name="QAAgent",
    system_message="""You are a senior Python engineer. Your task is to answer factual questions based on the data or chart images provided. You will generate Python code to analyze charts, images, or other formats to extract the answers. Here are the steps you need to follow:
    1. Load the pre-trained model ahmed-masry/ChartInstruct-LLama2
    2. When gathering information: preparing the device (GPU/CPU), processing the input image and question into tensors, generating the answer using beam search, decoding the output, and printing the result.
    3. When solving the task: - Generate the necessary Python code to solve the task. model = LlavaForConditionalGeneration.from_pretrained("ahmed-masry/ChartInstruct-LLama2", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/ChartInstruct-LLama2"). image = Image.open(image_path).convert("RGB");input_prompt = f"<image>\n Question: {input_text} Answer: ";inputs = processor(text=input_prompt, images=image, return_tensors="pt");inputs = {k: v.to(device) for k, v in inputs.items()};inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16);prompt_length = inputs['input_ids'].shape[1];generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512);output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] - Ensure the code is complete and ready to execute without requiring modifications from the user. Do not use raise, if-else, or try-catch to handle errors.
    4. Error handling: - If an error occurs, provide an updated script to fix the issue. Make sure to handle edge cases or potential issues in the data processing.
    5. Task completion: After completing the task and generating the required Python code. Your goal is to generate efficient Python code that leverages the google/deplot model to convert charts into tabular data. The code you generate should be complete, functional, and ready to execute. After generating the Python code, close the ```.
    6. Respond with "TERMINATE" to indicate that everything is done.
    """,
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
    code_execution_config=False
)

conversion_agent = AssistantAgent(
    name="ConversionAgent",
    system_message="""You are a senior Python engineer. Your task is to generate Python code that uses pre-trained models to analyze charts and generate underlying data. Specifically, you will use the google/deplot model to convert a chart into a data table. Here's how you should approach the task:
    1. When receiving a task: If the task involves generating data from a chart, use the pre-trained model google/deplot via the Pix2StructProcessor and Pix2StructForConditionalGeneration from the Hugging Face transformers library. The task is to take an image of a chart, process it, and output the underlying data as a table. You are expected to generate the Python code that uses this model to perform the task.
    2. When gathering information: You will load the image using Python's PIL.Image.open() and process it using Pix2StructProcessor. You should also prepare a description (e.g., "Generate underlying data table of the figure below:") to pass along with the image.
    3. When solving the task: Generate the Python code to load the model and processor, process the image, and use the model to generate the underlying data table. After obtaining the string returned from the model with max_new_tokens=512 (important), process the string further by data_string = data_string.replace('<0x0A>', '\n'); lines = data_string.split('\n'); has_title = "TITLE" in data_string; header_lines = [line.split(' | ') for line in lines if '|' in line][:2 if has_title else 1]; header_length = len(header_lines[-1]); lines = lines[2:] if has_title else lines[1:]; data = []; for line in lines:try:parts = line.split(' | ');data.append(parts);except ValueError:;pass; df = pd.DataFrame(data, columns=header_lines[-1]). Do not use raise, if-else, or try-catch to handle errors. The code should decode the model's output and print the generated data.
    4. Error handling: If there is an error in the code, provide a corrected version. Ensure the code works efficiently for the task at hand.
    5. Task completion: After completing the task and generating the required Python code. Your goal is to generate efficient Python code that leverages the google/deplot model to convert charts into tabular data. The code you generate should be complete, functional, and ready to execute. After generating the Python code, close the ```.
    6. Respond with "TERMINATE" to indicate that everything is done.
    """,
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
    code_execution_config=False
)

fact_checking_agent = AssistantAgent(
    name="FactCheckAgent",
    system_message="""You are a senior Python engineer. Your task is to answer factual questions based on the data or chart images provided. You will generate Python code to analyze charts, images, or other formats to extract the answers. Here are the steps you need to follow:
    1. Load the pre-trained model "khhuang/chartve" VisionEncoderDecoderModel and DonutProcessor
    2. When solving the task: - Generate the necessary Python code to solve the task.def format_query(sentence):return f"Does the image entails this statement: \"{sentence}\"?";query = format_query(CAPTION_SENTENCE);img = Image.open("chart_example_1.png");pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values;pixel_values = pixel_values.cuda();decoder_input_ids = processor.tokenizer(query, add_special_tokens=False, return_tensors="pt", max_length=510).input_ids.cuda();outputs = model(pixel_values, decoder_input_ids=decoder_input_ids);binary_entail_prob_positive = torch.nn.functional.softmax(outputs['logits'].squeeze()[-1,[2334, 49922]])[1].item(). Print output is output positive - Ensure the code is complete and ready to execute without requiring modifications from the user. Do not use raise, if-else, or try-catch to handle errors.
    3. Error handling: - If an error occurs, provide an updated script to fix the issue. Make sure to handle edge cases or potential issues in the data processing.
    4. Task completion: After generating the Python code, close the ```. After completing the task and generating the required Python code. Your goal is to generate efficient Python code that leverages the google/deplot model to convert charts into tabular data. The code you generate should be complete, functional, and ready to execute.
    5. Respond with "TERMINATE" to indicate that everything is done (outside code python).
    """,
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
    code_execution_config=False
)

planner = AssistantAgent(
    name="Planner",
    system_message="""
    You are a helpful AI assistant.
    Your tasks involve working with charts to provide accurate and insightful responses. Use your analytical and language skills to classify input questions into the following tasks:

    Answering Questions About Charts (Chart Question Answering):

    Converting Charts to Tables (Chart-to-Table Conversion):

    Verifying Chart Accuracy (Chart Fact-Checking):

    (Important) You will only respond by classifying your input into one of these three tasks or providing an answer related to these tasks. If the message does not fall into any of the three categories, return None.
    Input message: {input_message}
    """,
    llm_config={"config_list": config_list, "timeout": 60, "temperature": 0},
)

executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "MyAgent\\coding",
        "use_docker": False,
    },
)

def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        return planner
    
    if last_speaker is planner:
        if "Chart-to-Table Conversion" in messages[-1]["content"]:
            return conversion_agent
        elif "Chart Question Answering" in messages[-1]["content"]:
            return qa_agent
        elif "Chart Fact-Checking" in messages[-1]["content"]:
            return fact_checking_agent
        elif "None" in  messages[-1]["content"]:
            if (len(messages) <= 4):
                return boss
            else:
                listAgent = [conversion_agent, qa_agent, fact_checking_agent]
                for agent in listAgent:
                    if(messages[-4]["name"] == agent.name):
                        return agent

    elif last_speaker is conversion_agent:
        if "```python" in messages[-1]["content"]:
            return executor
        else:
            return conversion_agent
        
    elif last_speaker is qa_agent:
        if "```python" in messages[-1]["content"]:
            return executor
        else:
            return qa_agent
        
    elif last_speaker is fact_checking_agent:
        if "```python" in messages[-1]["content"]:
            return executor
        else:
            return fact_checking_agent
        
    elif last_speaker is executor:
        if "exitcode: 1" in messages[-1]["content"] or "Error" in messages[-1]["content"]:
            if "Chart-to-Table Conversion" in messages[1]["content"]:
                return conversion_agent
            elif "Chart Question Answering" in messages[1]["content"]:
                return qa_agent
            elif "Chart Fact-Checking" in messages[1]["content"]:
                return fact_checking_agent
        else:
            return boss
    elif last_speaker is boss:
        return planner
    else: 
        return "random"

groupchat = GroupChat(
    agents=[boss, planner, qa_agent, conversion_agent, fact_checking_agent, executor],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list, "timeout": 60, "temperature": 0})
image_path = r"D:\\Dai_hoc\\Cac_mon_hoc\\Ky_5\\Nhap_mon_du_lieu_lon\\CuoiKy\\MyAgent\\Chart\\chart_example_1.png"
message = f"Use image in {image_path} and caption: Share of Facebook Messenger users in the United States as of January 2018, by age group. Check if the caption matches the chart or not?"

boss.initiate_chat(
    manager,
    message=message,
)