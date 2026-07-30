# Image Caption Generator: ResNet50 + Transformer Decoder

A from-scratch image captioning model that pairs a frozen **ResNet50** vision
backbone with a hand-built **Transformer decoder** (masked self-attention,
cross-attention, feed-forward, all implemented from first principles — no
`tf.keras.layers.MultiHeadAttention`). Trained on **Flickr8k** and served
through a small Streamlit app.

This project is a direct extension of an earlier from-scratch
English→French Transformer: the text encoder is swapped out for a CNN, and
the decoder — masked self-attention, cross-attention, feed-forward, Add &
Norm, the warmup learning-rate schedule, the training loop — carries over
unchanged. The only new work is bridging a CNN's spatial feature map into
the sequence format the decoder's cross-attention expects.

## How it works

```
Image (224×224×3)
        │
        ▼
   ResNet50 (frozen, ImageNet weights)
        │  → (7, 7, 2048) spatial feature map
        ▼
   Reshape to (49, 2048)              "49 visual words," one per grid cell
        │
        ▼
   FFN: Dense(128, relu) → Dense(512, linear)
        │  → (49, 512), matching the decoder's d_model
        ▼
   ┌─────────────────────────────────────────┐
   │  Decoder × 3  (identical across layers,  │
   │  all cross-attending to the SAME image   │
   │  features — not progressively different  │
   │  ones)                                   │
   │                                          │
   │  masked self-attention (over caption      │
   │      tokens generated so far)            │
   │        → Add & Norm                       │
   │  cross-attention (Q = caption state,      │
   │      K/V = image features)                │
   │        → Add & Norm                       │
   │  feed-forward                             │
   │        → Add & Norm                       │
   └─────────────────────────────────────────┘
        │
        ▼
   Dense(vocab_size) → next-token logits
```

At inference, generation is autoregressive: start from `<sos>`, feed each
predicted token back in as the next input, and stop the moment `<eos>` is
predicted (or a max-length cap is hit).

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091
images, 5 captions each (~40,455 total image-caption pairs).

- **Split by image, not by caption** — 7,000 images for training, 500 for
  validation, 591 for testing. Splitting by image (rather than shuffling all
  40k captions together) matters here: since each image has 5 captions, a
  caption-level split would leak the same image into both train and test,
  quietly inflating validation/test scores.
- Captions tokenized (lowercased, punctuation split off, non-alphanumeric
  characters stripped), vocabulary built from the training split only
  (`min_freq=2`), with `<pad>`/`<sos>`/`<eos>`/`<unk>` special tokens.
- Captions padded to a fixed `MAX_LEN=30`.
- Images resized to 224×224 and run through
  `tf.keras.applications.resnet50.preprocess_input` (ImageNet channel
  normalization) inside the `tf.data` pipeline.

## Training

- Loss: masked sparse categorical crossentropy — `<pad>` positions excluded
  so the model isn't rewarded for "predicting" padding.
- Optimizer: Adam (`β1=0.9, β2=0.98, ε=1e-9`, matching the original
  Transformer paper's settings) with the paper's warmup learning-rate
  schedule (LR ramps up, then decays).
- Gradient clipping by global norm (1.0) for training stability.
- `@tf.function`-compiled training step for speed.
- ResNet50 is **frozen** (`trainable = False`) — only the projection FFN and
  the decoder are trained. With ~7k images, fine-tuning the full backbone
  risks overfitting fast and multiplies training cost for little benefit;
  ImageNet features already transfer well to "what objects/scenes are in
  this photo."
- 15 epochs.

## Key implementation challenges

### 1. Bridging a spatial feature map into a token sequence

The decoder's cross-attention was originally built to attend over a
sequence of encoder hidden states — one vector per *source token*. A CNN
doesn't naturally produce that. ResNet50's last convolutional block outputs
a `(7, 7, 2048)` feature map for a 224×224 input: a 7×7 spatial grid, 2048
channels deep at each cell.

The fix treats each of the 49 grid cells as if it were a "token": reshaping
`(7, 7, 2048) → (49, 2048)` turns the spatial grid into a sequence of 49
"visual words," each a 2048-dimensional feature vector describing that
region of the image. That's directly analogous to 49 source-sentence
tokens, each represented by an embedding vector — except here the
"embedding" comes from a CNN instead of a lookup table.

That still leaves a dimension mismatch: ResNet50's channel depth (2048)
doesn't match the decoder's `d_model` (512), and cross-attention requires
Q/K/V to operate in a shared dimensional space. A small feed-forward network
— `Dense(128, relu) → Dense(512, linear)` — projects each of the 49 vectors
from 2048 down to 512, independently at each spatial position. The result,
`(batch, 49, 512)`, is exactly the shape the existing `Decoder` class
already expected from a text encoder — so the decoder itself required zero
changes.

### 2. Making a stack of custom Keras classes serializable

`model.save('image_captioner.keras')` needs to persist not just weights but
the *architecture* — meaning Keras must be able to reconstruct every custom
`Layer`/`Model` subclass from a config dictionary on load. By default,
Keras has no way to know that `Decoder.__init__` needs `d_model`,
`num_heads`, and `d_ff` to rebuild itself — that requires a `get_config()`
method returning those values, on **every** custom class in the graph:
`self_attention`, `MultiHeadAttention`, `masked_self_attention`,
`masked_multi_head_attention`, `cross_attention`,
`multi_head_cross_attention`, `ffn`, `Decoder`, `ImageEncoder`, and
`ImageCaptioner` — ten classes in total.

Writing near-identical `get_config()` boilerplate ten times was avoided with
a single dynamic patcher:

```python
def patch_get_config(cls, arg_names):
    def get_config(self):
        config = super(cls, self).get_config()
        for arg in arg_names:
            if hasattr(self, arg):
                config[arg] = getattr(self, arg)
        return config
    cls.get_config = get_config

patch_get_config(ImageCaptioner, ['d_model', 'num_heads', 'd_ff', 'target_vocab_size'])
patch_get_config(ImageEncoder, ['d_model'])
patch_get_config(Decoder, ['d_model', 'num_heads', 'd_ff'])
patch_get_config(ffn, ['d_model', 'd_ff'])
patch_get_config(self_attention, ['d_model', 'num_heads'])
# ...and so on for the remaining attention classes
```

Each call attaches a `get_config` method to the class after the fact,
pulling whichever constructor arguments that class actually needs off the
instance (`d_model`, `num_heads`, etc., all stored as `self.x` in each
`__init__`) and merging them into the base config dict. This keeps every
class's serialization logic in one place instead of ten, and makes adding a
new custom layer later a one-line addition rather than another full
`get_config` method to write and maintain.

**This matters for deployment, too** — loading the saved model back
(e.g. in `model.py` for the Streamlit app) requires passing every custom
class through `custom_objects`, since Keras still needs the actual class
definitions available to reconstruct instances from the saved config:

```python
model = tf.keras.models.load_model(
    'image_captioner.keras',
    custom_objects={
        'ImageCaptioner': ImageCaptioner,
        'ImageEncoder': ImageEncoder,
        'Decoder': Decoder,
        'ffn': ffn,
        'self_attention': self_attention,
        'MultiHeadAttention': MultiHeadAttention,
        'masked_self_attention': masked_self_attention,
        'masked_multi_head_attention': masked_multi_head_attention,
        'cross_attention': cross_attention,
        'multi_head_cross_attention': multi_head_cross_attention,
    }
)
```

### 3. Vocabulary persistence

The model itself has no notion of "which integer means which word" — that
mapping lives entirely in the `Vocab` object built during preprocessing.
Since the trained model needs to be usable outside the notebook that
trained it, `vocab.itos` (index → string) and `vocab.stoi` (string → index)
are serialized to `vocab.json` alongside the model weights. Any downstream
consumer (the Streamlit app included) loads both files together — the model
to generate token ids, the vocab to turn those ids back into words.

## Inference

Greedy autoregressive decoding, identical in spirit to the text-to-text
Transformer this project builds on: start from `<sos>`, run one decoding
step at a time feeding each prediction back in as the next input, and stop
as soon as `<eos>` is generated (capped at `max_len` steps as a safety net).
The image's 49 projected feature vectors are computed once per image and
reused at every decoding step — only the caption side grows.

## Project structure

```
├── Image_caption_generator_using_encoder_decoder.ipynb   # training notebook (this repo's core)
├── image_captioner.keras       # saved model (architecture + weights)
├── vocab.json                   # itos / stoi mappings
├── model.py                     # loads the saved model + vocab, exposes a captioning function
├── utils.py                     # image preprocessing / helper functions shared by model.py and the app
└── streamlit_app.py              # frontend: upload an image, get a generated caption
```

`model.py` is responsible for the `custom_objects`-aware load shown above
plus a `generate_caption(image)` function mirroring the notebook's
inference loop. `utils.py` handles turning an uploaded image (e.g. a
Streamlit `UploadedFile`) into the same `224×224`, ResNet50-preprocessed
tensor the model was trained on. `streamlit_app.py` wires the two together
behind a simple upload-and-display UI.

## Running it

**Training** (Colab, GPU recommended):
1. Open the notebook, provide a `kaggle.json` API token, run top to bottom.
2. Trained weights save to `image_captioner.keras`, vocabulary to
   `vocab.json`.

**Local app**:
```bash
pip install tensorflow streamlit pillow
streamlit run streamlit_app.py
```
Place `image_captioner.keras` and `vocab.json` in the same directory (or
point `model.py` at wherever you saved them), then upload an image in the
browser UI to get a generated caption.

## Known limitations / possible extensions

- The 49 image regions carry no positional encoding — the decoder can't
  currently distinguish "top-left of the image" from "bottom-right." A 2D
  positional encoding over the 7×7 grid (analogous to the 1D encoding used
  for caption tokens) would let the model reason about spatial layout, e.g.
  "a dog **behind** a fence" vs. "a dog **in front of** a fence."
- No self-attention among the 49 image regions before cross-attention — the
  decoder attends directly to raw (projected) CNN features rather than
  features that have already been contextualized against each other. Adding
  a lightweight self-attention block over the image sequence (the same
  `Encoder` block used in the text-to-text version) could let the model
  reason about relationships between objects before generating each word.
- ResNet50 is frozen; fine-tuning the later conv blocks at a low learning
  rate, once the decoder has converged, sometimes helps if you have more
  compute budget to spend.
- Greedy decoding only — beam search would generally push caption quality
  up further at the cost of extra inference compute.
