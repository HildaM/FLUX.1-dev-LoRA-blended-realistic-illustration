---
tags:
- text-to-image
- stable-diffusion
- lora
- diffusers
- image-generation
- flux
- safetensors
widget:
- text: >-
    A cartoon-style blonde European-American woman wearing sunglasses stood in front of the triumphant door to take a selfie, the upper body, the art style combines reality and illustration elements.,
  output:
    url: >-
      images/example1.png
- text: >-
    A cartoon style European woman wearing glasses is eating a table of seafood,including lobster,oysters,and other shellfish,in a well lit modern restaurant. The background of the restaurant is very blurry,and she is holding the utensils ready to eat. There is a glass of red wine and various dishes on the table. The illustrations contrast with the real food and environment,creating a unique mixed media effect and high angle perspective. The artistic style blends elements of reality and illustration.,
  output:
    url: >-
      images/example2.png
- text: >-
    A man wearing a hat, sunglasses, white t-shirt, black vest, black shorts, and white sneakers squatted on a wooden path. Next to him was a cartoon style woman wearing a gray top, black skirt, and white shoes. She raised her hands above her head and leaned against a textured geometric background wall. A cartoon Doberman Inu sat next to the cartoon woman, wearing a black harness and a frisbee at her feet. The artistic style blends elements of reality and illustration.
  output:
    url: >-
      images/example3.png
- text: >-
    In a car,a cartoon style European blonde woman sits in the passenger seat,while a man sits in the driver's seat,holding the steering wheel with one hand. A stylish modern car with a black steering wheel,brown leather seats,and a background showcasing flowers and water bodies. On a sunny day,this car can be seen from the driver's side window. The artistic style blends elements of reality and illustration.,
  output:
    url: >-
      images/example4.png
- text: >-
    A cartoon style European man opens his hands and takes a selfie under the Sydney Opera House,blending elements of reality and illustration in his artistic style.,
  output:
    url: >-
      images/example5.png
- text: >-
    A cartoon-style blonde European and American woman wearing sunglasses standing in front of Sofia Cathedral to take a selfie,the upper body,the art style combines reality and illustration elements.,
  output:
    url: >-
      images/example6.png
- text: >-
    A cartoon-style Indian girl,dressed in traditional Indian costumes and veil,takes a selfie in front of the Taj Mahal,artistic style blends reality and illustration elements.,upper_body,
  output:
    url: >-
      images/example7.png
- text: >-
    A cartoon-style black man, shirtless, wearing sunglasses, gold necklace, taking a selfie on a cruise ship, behind the Statue of Liberty, artistic style blends reality and illustration elements.
  output:
    url: >-
      images/example8.png
- text: >-
    A modern outdoor poolside scene,with a blonde European and American woman lying on an orange willow lounge chair. She is depicted as a cartoon character wearing a bikini and white sandals. The background features beige concrete walls,glass doors with curtains,and tropical palm trees. The sunlight casts shadows on the tile floor and building facade,creating a fresh and warm weather and a relaxed and peaceful atmosphere. The artistic style blends elements of reality and illustration.,
  output:
    url: >-
      images/example9.png
- text: >-
    In front of a graffiti filled street wall in reality,a cartoon style black man wearing a black jacket,a gold necklace,and a baseball cap is taking a selfie with his phone. The artistic style blends elements of reality and illustration.,
  output:
    url: >-
      images/example10.png
- text: >-
    A cartoon style European woman wearing glasses is eating a table of seafood,including lobster,oysters,and other shellfish,in a well lit modern restaurant. The background of the restaurant is very blurry,and she is holding the utensils ready to eat. There is a glass of red wine and various dishes on the table. The illustrations contrast with the real food and environment,creating a unique mixed media effect and high angle perspective. The artistic style blends elements of reality and illustration.,
  output:
    url: >-
      images/example11.png
- text: >-
    Against the backdrop of the Eiffel Tower, a cartoon style European woman wearing a delicate white floral dress stands there, with the iconic building of the tower clearly visible under the azure sky, capturing the romantic charm of Paris. When she takes photos against this stunning background, her flowing hair adds a dreamy atmosphere, and the artistic style blends reality and illustration elements.
  output:
    url: >-
      images/example12.png
- text: >-
    A cartoon-style Indian woman,praying with her hands folded under a huge Buddha statue,dressed in traditional Indian clothing,wearing a veil and beads on her hands,artistic style blends reality and illustration elements.,
  output:
    url: >-
      images/example13.png
- text: >-
    A European and American man wearing a hat sat at the dining table,using his smartphone to capture another cartoon style blonde European and American woman across from him. On the marble table,there are various colorful foods and desserts,including fruits,pastries,and drinks. The background is a large window,where trees can be seen from the outside,and the natural light is bright,creating a relaxed dining environment. The entire scene creatively blends elements of reality and illustration. The artistic style blends elements of reality and illustration.,
  output:
    url: >-
      images/example14.png
- text: >-
    On the tennis court,a cartoon style American cheerleader dances in front of an indoor basketball court,posing in a dance style that emphasizes sports. The characteristics,blurred background,and artistic style blend elements of reality and illustration.,upper_body,
  output:
    url: >-
      images/example15.png
base_model: black-forest-labs/FLUX.1-dev
instance_prompt: artistic style blends reality and illustration elements.
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
language:
- en
pipeline_tag: text-to-image
library_name: diffusers
---
# FLUX.1-dev-LoRA-blended-realistic-illustration

This is a LoRA (Vector Journey) trained on FLUX.1-dev for blended realistic illustration by [Muertu](https://www.shakker.ai/userpage/562c8ddb15e147c9b1c31878f865c901/publish), the front character is in illustration style, while the background is realistic.
We will discolse more details about how to prepare training dataset soon!

<div class="container">
  <img src="./images/poster.jpeg" width="1024"/>
</div>


## Showcases
<Gallery />

## Trigger words

You should use `artistic style blends reality and illustration elements.` to trigger the image generation. The recommended scale is `1.0` in diffusers, but for ComfyUI or WebUI, the scale may be different.

## Inference

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-blended-realistic-illustration", weight_name="FLUX-dev-lora-blended_realistic_illustration.safetensors")
pipe.fuse_lora(lora_scale=1.0)
pipe.to("cuda")

prompt = "A peaceful outdoor scene,surrounded by lush green leaves and vibrant vegetation on the stone road. A cartoon style man sits on white clothes,gently caressing two animated cats,one orange and one white. Soft sunlight creates shadows,simple walls and pebble floors create a peaceful atmosphere,blending reality and animation elements,and natural and procedural characters. The artistic style blends elements of reality and illustration."

image = pipe(prompt, 
             num_inference_steps=24, 
             guidance_scale=3.5,
             width=768, height=1024,
            ).images[0]
image.save(f"example.png")
```


## Online Inference

You can also download this model at [Shakker AI](https://www.shakker.ai/modelinfo/4ea23c6ad148462589ea42e4eeac9897?from=personal_page), where we provide an online interface to generate images.


## Acknowledgements
This model is trained by our copyrighted users [Muertu](https://www.shakker.ai/userpage/562c8ddb15e147c9b1c31878f865c901/publish). We release this model under permissions. The model follows [flux-1-dev-non-commercial-license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) and the generated images are also non commercial.
