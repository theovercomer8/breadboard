from json import JSONDecoder, JSONEncoder
import sys
import getopt
import time
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import hashlib
import inspect
import math
import numpy as np
import open_clip
import pickle
import time
from dataclasses import dataclass
from blip.models.blip import blip_decoder, BLIP_Decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from typing import List



# All objects we find
json_found = []  
# raw_decode expects byte1 to be part of a JSON, so remove whitespace from left
# stdin = sys.stdin.read().lstrip()
decoder = JSONDecoder()
encoder = JSONEncoder()

git_fail_phrases = 'a sign that says,writing that says,that says'
git_pass = False
blip_pass = False
cap_length = 75
existing = 'ignore'
clip_beams = 8
clip_min = 30
clip_max = 50
clip_v2 = False
clip_use_flavor = False
clip_max_flavors = 4
clip_use_artist = False
clip_use_medium = False
clip_use_movement = False
clip_use_trending = False
ignore_tags = ''
replace_class = False
sub_class = ''
sub_name = ''
folder_tag = False
folder_tag_levels = 1
uniquify_tags = False
write_to_file = False
use_filename = False

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None

def get_parent_folder(filepath, levels=1):
    common = os.path.split(filepath)[0]
    paths = []
    for i in range(int(levels)):
        split = os.path.split(common)
        common = split[0]
        paths.append(split[1])
    return paths


def git_caption(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    pixel_values = pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=150)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


        

BLIP_MODELS = {
    'base': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
    'large': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
}

@dataclass 
class Config:
    # models can optionally be passed in directly
    blip_model: BLIP_Decoder = None
    clip_model = None
    clip_preprocess = None

    # blip settings
    blip_image_eval_size: int = 384
    blip_model_type: str = 'large' # choose between 'base' or 'large'
    blip_offload: bool = False

    # clip settings
    clip_model_name: str = 'coca_ViT-L-14/mscoco_finetuned_laion2B-s13B-b90k'
    clip_model_path: str = None

    # interrogator settings
    cache_path: str = 'cache'
    chunk_size: int = 2048
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    flavor_intermediate_count: int = 2048
    quiet: bool = False # when quiet progress bars are not shown

class Interrogator():
    def __init__(self, config: Config):
        self.config = config
        self.device = device

        if blip_pass:
            if config.blip_model is None:
                if not config.quiet:
                    print("Loading BLIP model...")
                blip_path = os.path.dirname(inspect.getfile(blip_decoder))
                configs_path = os.path.join(os.path.dirname(blip_path), 'configs')
                med_config = os.path.join(configs_path, 'med_config.json')
                blip_model = blip_decoder(
                    pretrained=BLIP_MODELS[config.blip_model_type],
                    image_size=config.blip_image_eval_size, 
                    vit=config.blip_model_type, 
                    med_config=med_config
                )
                blip_model.eval()
                blip_model = blip_model.to(config.device)
                self.blip_model = blip_model
            else:
                self.blip_model = config.blip_model

        if clip_use_movement or clip_use_artist or clip_use_flavor or clip_use_medium or clip_use_trending:
            self.load_clip_model()

    def load_clip_model(self):
        start_time = time.time()
        config = self.config

        if config.clip_model is None:
            if not config.quiet:
                print("Loading CLIP model...")

            clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained=clip_model_pretrained_name, 
                precision='fp16' if config.device == 'cuda' else 'fp32',
                device=config.device,
                jit=False,
                cache_dir=config.clip_model_path
            )
            self.clip_model.to(config.device).eval()
        else:
            self.clip_model = config.clip_model
            self.clip_preprocess = config.clip_preprocess
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])

        raw_artists = _load_list(config.data_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        if clip_use_artist:
            self.artists = LabelTable(artists, "artists", self.clip_model, self.tokenize, config)
        
        if clip_use_flavor:
            self.flavors = LabelTable(_load_list(config.data_path, 'flavors.txt'), "flavors", self.clip_model, self.tokenize, config)
        
        if clip_use_medium:
            self.mediums = LabelTable(_load_list(config.data_path, 'mediums.txt'), "mediums", self.clip_model, self.tokenize, config)
        
        if clip_use_movement:
            self.movements = LabelTable(_load_list(config.data_path, 'movements.txt'), "movements", self.clip_model, self.tokenize, config)
        
        if clip_use_trending:
            self.trendings = LabelTable(trending_list, "trendings", self.clip_model, self.tokenize, config)

        end_time = time.time()
        if not config.quiet:
            print(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")

    def generate_blip_caption(self, pil_image: Image) -> str:
        if self.config.blip_offload:
            self.blip_model = self.blip_model.to(self.device)
        size = self.config.blip_image_eval_size
        gpu_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, 
                sample=False, 
                num_beams=clip_beams, 
                max_length=clip_max, 
                min_length=clip_min
            )
        if self.config.blip_offload:
            self.blip_model = self.blip_model.to("cpu")
        return caption[0]

    def image_to_features(self, image: Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    
    def interrogate(self, caption: str, image: Image) -> str:
        image_features = self.image_to_features(image)

        # flaves = self.flavors.rank(image_features, self.config.flavor_intermediate_count)
        # best_medium = self.mediums.rank(image_features, 1)[0]
        # best_artist = self.artists.rank(image_features, 1)[0]
        # best_trending = self.trendings.rank(image_features, 1)[0]
        # best_movement = self.movements.rank(image_features, 1)[0]

        best_prompt = caption
        best_sim = self.similarity(image_features, best_prompt)

        def check(addition: str) -> bool:
            nonlocal best_prompt, best_sim
            prompt = best_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if sim > best_sim:
                best_sim = sim
                best_prompt = prompt
                return True
            return False

        def check_multi_batch(opts: List[str]):
            nonlocal best_prompt, best_sim
            prompts = []
            for i in range(2**len(opts)):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

            t = LabelTable(prompts, None, self.clip_model, self.tokenize, self.config)
            best_prompt = t.rank(image_features, 1)[0]
            best_sim = self.similarity(image_features, best_prompt)

        batch = []

        if clip_use_artist:
            batch.append(self.artists.rank(image_features,1)[0])
        if clip_use_flavor:
                best_flavors = self.flavors.rank(image_features, self.config.flavor_intermediate_count)
                extended_flavors = set(best_flavors)
                for _ in tqdm(range(clip_max_flavors), desc="Flavor chain", disable=self.config.quiet):
                    best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
                    flave = best[len(best_prompt) + 2:]
                    if not check(flave):
                        break
                    if _prompt_at_max_len(best_prompt, self.tokenize):
                        break
                    extended_flavors.remove(flave)
        if clip_use_medium:
            batch.append(self.mediums.rank(image_features, 1)[0])
        if clip_use_trending:
            batch.append(self.trendings.rank(image_features, 1)[0])
        if clip_use_movement:
            batch.append(self.movements.rank(image_features, 1)[0])

        check_multi_batch(batch)
        tags = best_prompt.split(",")

        return tags
        # check_multi_batch([best_medium, best_artist, best_trending, best_movement])

        # extended_flavors = set(flaves)
        # for _ in tqdm(range(max_flavors), desc="Flavor chain", disable=self.config.quiet):
        #     best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
        #     flave = best[len(best_prompt)+2:]
        #     if not check(flave):
        #         break
        #     if _prompt_at_max_len(best_prompt, self.tokenize):
        #         break
        #     extended_flavors.remove(flave)

        # return best_prompt

    def rank_top(self, image_features: torch.Tensor, text_array: List[str]) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()


class LabelTable():
    def __init__(self, labels:List[str], desc:str, clip_model, tokenize, config: Config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels
        self.tokenize = tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = None
        if config.cache_path is not None and desc is not None:
            os.makedirs(config.cache_path, exist_ok=True)
            sanitized_name = config.clip_model_name.replace('/', '_').replace('@', '_')
            cache_filepath = os.path.join(config.cache_path, f"{sanitized_name}_{desc}.pkl")
            if desc is not None and os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        if data.get('hash') == hash:
                            self.labels = data['labels']
                            self.embeds = data['embeds']
                    except Exception as e:
                        print(f"Error loading cached table {desc}: {e}")

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                text_tokens = self.tokenize(chunk).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if cache_filepath is not None:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump({
                        "labels": self.labels, 
                        "embeds": self.embeds, 
                        "hash": hash, 
                        "model": config.clip_model_name
                    }, f)

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]
    
    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int=1) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def _load_list(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def _merge_tables(tables: List[LabelTable], config: Config) -> LabelTable:
    m = LabelTable([], None, None, None, config)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m

def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0

def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text

ci:Interrogator = None

def main(argv):
    global ci,processor,model,git_fail_phrases, git_pass, blip_pass, cap_length, existing, clip_beams, clip_min, clip_max, clip_v2, clip_use_flavor, clip_max_flavors, clip_use_artist, clip_use_medium, clip_use_movement, clip_use_trending, ignore_tags, replace_class, sub_class, sub_name, folder_tag, folder_tag_levels, uniquify_tags, write_to_file, use_filename
    opts, args = getopt.getopt(argv,"",["git_fail_phrases=","git_pass","blip_pass","cap_length=","existing=","clip_beams=","clip_min=","clip_max=","clip_v2","clip_use_flavor","clip_max_flavors=","clip_use_artist","clip_use_medium","clip_use_movement","clip_use_trending","ignore_tags=","replace_class","sub_class=","sub_name=", "folder_tag", "folder_tag_levels=", "uniquify_tags","write_to_file","use_filename"])
    for opt, arg in opts:
        if opt == '--git_fail_phrases':
            git_fail_phrases = arg
        elif opt == '--git_pass':
            git_pass = True
        elif opt == '--blip_pass':
            blip_pass = True
        elif opt == '--cap_length':
            cap_length = int(arg)
        elif opt == '--existing':
            existing = arg
        elif opt == '--clip_beams':
            clip_beams = int(arg)
        elif opt == '--clip_min':
            clip_min = int(arg)
        elif opt == '--clip_max':
            clip_max = int(arg)
        elif opt == '--clip_v2':
            clip_v2 = True
        elif opt == '--clip_use_flavor':
            clip_use_flavor = True
        elif opt == '--clip_max_flavors':
            clip_max_flavors = int(arg)
        elif opt == '--clip_use_artist':
            clip_use_artist = True
        elif opt == '--clip_use_medium':
            clip_use_medium = True
        elif opt == '--clip_use_movement':
            clip_use_movement = True
        elif opt == '--clip_use_trending':
            clip_use_trending = True
        elif opt == '--ignore_tags':
            ignore_tags = arg
        elif opt == '--replace_class':
            replace_class = True
        elif opt == '--sub_class':
            sub_class = arg
        elif opt == '--sub_name':
            sub_name = arg
        elif opt == '--folder_tag':
            folder_tag = True
        elif opt == '--folder_tag_levels':
            folder_tag_levels = int(arg)
        elif opt == '--uniquify_tags':
            uniquify_tags = True
        elif opt == '--write_to_file':
            write_to_file = True
        elif opt == "--use_filename":
            use_filename = True


    ##### INIT HERE. SEND --READY-- WHEN READY FOR INPUT
    if git_pass:
        processor = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps")
        model.to(device)
    
    if clip_use_movement or clip_use_artist or clip_use_flavor or clip_use_medium or clip_use_trending or blip_pass:
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai",
                             quiet=False))

    print("--READY--", flush=True)
    isReady = input()
    if(isReady.rstrip() == "check"):
        print("ready")

    while True:
        try:
            c = input()
            Message = c.rstrip()

            parsed_json = decoder.decode(Message)
            if 'finished' in parsed_json:
                print('--FINISHED--')
                sys.exit()
            process_img(parsed_json)

            # Flush all the print statements from console
            sys.stdout.flush()
            time.sleep(0.1)
        except EOFError:
            time.sleep(0.5)
        
def process_img(obj):
    # Load image
    img_path = obj['path']
    img = Image.open(img_path).convert('RGB')

    # Get existing caption
    existing_caption = ''
    if 'caption' in obj:
        existing_caption = obj['caption']

    # Get caption from filename if empty
    if existing_caption == '' and use_filename:
        path = os.path.split(img_path)[1]
        path = os.path.splitext(path)[0]
        existing_caption = ''.join(c for c in path if c.isalpha() or c in [" ", ","])
    
    
    # Create tag list
    out_tags = []
    new_caption = ''
   
    # 1st caption pass: GIT
    if git_pass:
        new_caption = git_caption(img)
        print('Got git caption: ',new_caption)
        # Check if caption fails from list of not-allowed phrases
        if blip_pass and any(f in new_caption for f in git_fail_phrases.split(',')):
            # Fail git caption
            new_caption = ''

    # 2nd caption pass: BLIP (if failed)
    if blip_pass and new_caption == '':
        new_caption = ci.generate_blip_caption(img)



    # Add enabled CLIP flavors to tag list
    if clip_use_artist or clip_use_flavor or clip_use_medium or clip_use_movement or clip_use_trending:
        tags = ci.interrogate(new_caption,img)
        for tag in tags:
            out_tags.append(tag)
    else:
        for tag in new_caption.split(","):
            out_tags.append(tag)


    # Add parent folder to tag list if enabled
    if folder_tag:
        folder_tags = get_parent_folder(img_path,folder_tag_levels)
        for tag in folder_tags:
            out_tags.append(tag)

    # Remove duplicates, filter dumb stuff
    # chars_to_strip = ["_\\("]
    unique_tags = []
    tags_to_ignore = []
    if ignore_tags != "" and ignore_tags is not None:
        si_tags = ignore_tags.split(",")
        for tag in si_tags:
            tags_to_ignore.append(tag.strip)

    if uniquify_tags:
        for tag in out_tags:
            if not tag in unique_tags and not "_\(" in tag and not tag in ignore_tags:
                unique_tags.append(tag.strip())
    else:
         for tag in out_tags:
            if not "_\(" in tag and not tag in ignore_tags:
                unique_tags.append(tag.strip())

    existing_tags = existing_caption.split(",")

    # APPEND/PREPEND/OVERWRITE existing caption based on options
    if existing == "prepend" and len(existing_tags):
        new_tags = existing_tags
        for tag in unique_tags:
            if not tag in new_tags or not uniquify_tags:
                new_tags.append(tag)
        unique_tags = new_tags

    if existing == 'append' and len(existing_tags):
        for tag in existing_tags:
            if not tag in unique_tags or not uniquify_tags:
                unique_tags.append(tag)

    if existing == 'copy' and existing_caption:
        for tag in existing_tags:
            unique_tags.append(tag.strip())

    # Construct new caption from tag list
    caption_txt = ", ".join(unique_tags)

    if replace_class and sub_name is not None and sub_class is not None:
        # Find and replace "a SUBJECT CLASS" in caption_txt with subject name
        if f"a {sub_class}" in caption_txt:
            caption_txt = caption_txt.replace(f"a {sub_class}", sub_name)

        if sub_class in caption_txt:
            caption_txt = caption_txt.replace(sub_class, sub_name)

    tags = caption_txt.split(" ")
    if cap_length != 0 and len(tags) > cap_length:
            tags = tags[0:cap_length]
            tags[-1] = tags[-1].rstrip(",")
    caption_txt = " ".join(tags)

    # Write caption file
    if write_to_file:
        with open(os.path.splitext(img_path)[0] + '.txt', "w", encoding="utf8") as file:
                    file.write(caption_txt)

    # Return caption text to Breadboard
    print('PROCESSED: {0} | NEW CAPTION:{1}'.format(img_path,caption_txt))



if __name__ == "__main__":
   main(sys.argv[1:])
