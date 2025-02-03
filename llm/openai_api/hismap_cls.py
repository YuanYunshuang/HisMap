import glob
import json

import cv2
from openai import OpenAI
import base64
import requests
import os
import openai

os.environ['OPENAI_API_KEY'] = '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'
templete1 = """
                    On the left side is some examples of the symbols of a historical map including the class Wood 
                    (triangle- or circle-like small symbols with the bottom open and several dots on the bottom right side of the symbol)., 
                    Settlement (area with scattered points), Grass (area with scattered double dots), 
                    Water (blue area, note that blue lines are not included as water class). 
                    On the right side is a image of historical map patch. The examples on the left might appear in larger 
                    size on the right image. For the right image, please answer the following question with yes or no: 
                    1. Does the image contain Wood? 
                    2. Does the image contain Grass? 
                    3. Does the image contain Settlement? 
                    4. Does the image contain Water?
"""

def load_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Function to set memory
def set_memory(user_id, memory_data):
    response = openai.Memory.create(
        user_id=user_id,
        memory=memory_data
    )
    return response


# Function to retrieve memory
def get_memory(user_id):
    response = openai.Memory.retrieve(user_id=user_id)
    return response


client = OpenAI()

def get_cls_info(image_path):
    image = load_image_to_base64(image_path)
    response = client.chat.completions.create(
      model="chatgpt-4o-latest",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text":
                  """
                   On the left side is some examples of the symbols of a historical map including the class 
                   Wood, Settlement, Grass, Water (blue lines are not counted as water class). 
                   On the right side is a image of historical map patch. The examples on the left might appear in larger size on the right image. 
                   For the right image, please answer the following question with Yes or No and give reasons for the answer: 
                    1. Does the image contain Wood? 
                    2. Does the image contain Grass? 
                    3. Does the image contain Settlement? 
                    4. Does the image contain Water?
                    
                    Formatting the answer with the following stucture:
                    1. **Wood?** [Yes/No] : [reason]
                    2. **Grass?** [Yes/No] : [reason]  
                    3. **Settlement?** [Yes/No] : [reason]  
                    4. **Water?** [Yes/No] : [reason]
                  """,
            },
            {
              "type": "image_url",
              "image_url": {
                "url":  f"data:image/png;base64,{image}"
              },
            },
          ],
        }
      ],
    )

    return response


def parse_resonse(response):
    content = response.choices[0].message.content
    try:
        inds = [content.index(f'{i:d}.') for i in range(1, 5)]
        resusts = [
            content[inds[0]:inds[1]].strip(),
            content[inds[1]:inds[2]].strip(),
            content[inds[2]:inds[3]].strip(),
            content[inds[3]:].strip()
        ]

        onehot_cls = []
        additional_info = []
        for res in resusts:
            if 'Yes' in res:
                onehot_cls.append(1)
                additional_info.append(res[res.index('Yes'):])
            else:
                onehot_cls.append(0)
                additional_info.append(res[res.index('No'):])
    except:
        return None, content
    return onehot_cls, additional_info


if __name__ == '__main__':
    files = sorted(glob.glob("/home/yuan/data/HisMap/LGLN/dataset/llm_label/prompt_img/197x/*.jpg"))
    start_idx = 759
    for i, f in enumerate(files[start_idx:]):
        print(i + start_idx)
        llm_res = {}
        response = get_cls_info(f)
        onehot_cls, additional_info = parse_resonse(response)
        llm_res[os.path.basename(f)] = {
            'cls': onehot_cls,
            'info': additional_info,
        }
        # print(json.dumps(llm_res, indent=4))
        # image = cv2.imread(f)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        with open(f"/home/yuan/code/HisMap/data/llm/{os.path.basename(f).split('.')[0]}.json", "w") as fh:
            json.dump(llm_res, fh, indent=4)

    data = {
        "3821_1974_0_384_0110.jpg": '1. **Does the image contain Wood?**  \n   **No.** There are no symbols resembling open triangles or circles with dots on the bottom right that indicate wood.  \n\n2. **Does the image contain Grass?**  \n   **Yes.** Scattered double dots indicating grass are visible in the right-side image.  \n\n3. **Does the image contain Settlement?**  \n   **Yes.** The equally distributed points with black squares indicating buildings are visible, representing settlement.  \n\n4. **Does the image contain Water?**  \n   **Yes.** A blue area is visible, which corresponds to the water class.',
    }