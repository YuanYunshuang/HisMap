from openai import OpenAI
import base64
import requests
import os

os.environ['OPENAI_API_KEY'] = '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'


def load_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


client = OpenAI()

def initialize(image_path):
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
                  On the left side is the legend of an old map, on the right side is a patch of the old map. Answer the following questions with yes or no:
                    1. Does the map patch contain Laubwald?
                    2. Does the map patch contain Nadelwald?
                    3. Does the map patch contain Mischwald?
                    4. Does the map patch contain Wiese?
                    5. Does the map contain Bruchfeld?
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

    print(response.choices[0])


def get_response(image_path):
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
                  Answer the same questions with yes or no for the new given image:
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

    print(response.choices[0])



if __name__ == '__main__':
    initialize("/home/yuan/Downloads/hismap1.png")