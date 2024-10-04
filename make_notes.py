from typing import List
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer


def create_prompt(transcript: str):
    prompt = f"""You are reading lecture transcripts and will take detailed notes from each one. 
    
CONTEXT:
- Each lecture transcript usually consists of 2 professors talking with each other, but the transcript doesn't differentiate between them.
- Some lectures consist of running through examples.
- In your notes, be sure to highlight key concepts.

GUIDELINES
- DO NOT start your response with a greeting or confirmation. Start your response by starting the notes.

LECTURE TRANSCRIPT
{transcript}"""

    return prompt


def apply_llama_chat_template(prompts: List[str], tokenizer, system_prompt=None):
    if system_prompt is None:
        system_prompt = "You are an A+ student who is great at taking notes and highlighting and explaining academic concepts."
        
    prompt_list = []
    for prompt in prompts:
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        prompt_list.append(tokenizer.apply_chat_template(messages, tokenize=True))
    
    return prompt_list


if __name__ == '__main__':
    
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=1200
    )

    model_name = 'casperhansen/llama-3-70b-instruct-awq'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
            model=model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=.93,
            max_model_len=4096,
            dtype="auto",
        )

    df = pd.read_csv('/home/nate/whisper-lecture-transcriptions/KBAI_lessons_transcriptions.csv')
    df['prompt'] = df['transcription'].apply(create_prompt)

    prompt_token_ids = apply_llama_chat_template(
        prompts=df['prompt'].tolist(),
        tokenizer=tokenizer,
    )

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids
    )
    
    df['notes'] = [val.outputs[0].text for val in outputs]
    
    df.to_csv('KBAI_lecture_notes.csv')