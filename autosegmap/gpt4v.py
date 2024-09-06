# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#https://github.com/facebookresearch/open-eqa/blob/main/openeqa/utils/openai_utils.py#L61 해당 링크

import base64
import os
from typing import List, Optional

import cv2
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pandas as pd 


# def prepare_openai_messages(content: str):
#     return [{"role": "user", "content": content}]

def load_prompt(tgt_path):
    with open(tgt_path, "r") as f:
        return f.read().strip()

def load_img_prompt(tgt_img_path, image_size):
    frame = cv2.imread(tgt_img_path)
    if image_size:
        factor = image_size / max(frame.shape[:2])
        frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
    _, buffer = cv2.imencode(".png", frame) #추후에 파일형식 바꿔야할 듯?
    frame = base64.b64encode(buffer).decode("utf-8")
    img_msg = {
            "image_url": {"url": f"data:image/png;base64,{frame}"},
            "type": "image_url",
        }
    
    return img_msg


def load_text_prompt(text_prompt): 
    prompt = {
        "text": text_prompt,
        "type": "text"
    }
    return prompt

def prepare_openai_vision_messages( #gpt가 입력받는 것들. 
    prompt_path,
    img1_path,
    img2_path,
    text_prompt,
    image_size: Optional[int] = 512,
):

    prefix = load_prompt(prompt_path)    #프롬프트 경로
    content = []
    if prefix:
        content.append({"text": prefix, "type": "text"})
    
    content.append({"text": "[img1]", "type": "text"}) #받을 이미지
    content.append(load_img_prompt(img1_path, image_size)) 
    
    content.append({"text": "[img2]", "type": "text"}) #받을 이미지 이게 총 4pose인 4장을 넣어야함. 
    content.append(load_img_prompt(img2_path, image_size))

    content.append({"text": "[prompt]", "type": "text"}) #프롬프트
    content.append(load_text_prompt(text_prompt))

 
    return [{"role": "user", "content": content}]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) #이게 뭔지?
def call_openai_api(
    api_key: str,
    messages: list,
    model: str = "gpt-4",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = openai.OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


if __name__ == "__main__":
    #***api_key = api_key 변경하기.***
    img_dir = '/workspace/dataset/pickrich/rich100/'

    rich100= pd.read_csv('/workspace/dataset/pickrich/rich100.csv')

    # text_prompt = "Topology optimized structure" ##
    
    prompt_path = "prompts/prompt1.txt" #txt또는 그냥 txt_prompt로 바로 박기
    results = []

    for index, row in rich100.iterrows():
        img1_path = img_dir + row['image1'].replace('train/', '') #2장씩~ 내 생각 obj의 4pose를 다 넣어도 될듯? 또는 obj set을 다 넣어도 될것 같다.
        img2_path = img_dir + row['image2'].replace('train/', '')
        text_prompt = row['caption']

        messages = prepare_openai_vision_messages(prompt_path, img1_path, img2_path,text_prompt)

        model = "gpt-4o-2024-05-13"
        output = call_openai_api(api_key, messages, model=model, seed=1234, max_tokens=1024, temperature=0.2)
        print(output)

        results.append({
        'index': index,
        'image1': img1_path,
        'image2': img2_path,
        'caption': text_prompt,
        'output': output
        })
        
        gpt4results = pd.DataFrame(results)
        gpt4results.to_csv('gpt4_results.csv', index=False)
        
  
