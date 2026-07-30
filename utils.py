import json
import tensorflow as tf
from model import ImageCaptioner, Decoder, ImageEncoder, ffn, masked_multi_head_attention, multi_head_cross_attention

def load_resources(model_path, vocab_path):
    # 1. Load Vocab
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    stoi = vocab_dict['stoi']
    itos = vocab_dict['itos']
    
    # 2. Load Model (Custom objects help Keras map the strings to your classes)
    custom_objs = {
        'ImageCaptioner': ImageCaptioner, 'ImageEncoder': ImageEncoder, 
        'Decoder': Decoder, 'ffn': ffn, 
        'masked_multi_head_attention': masked_multi_head_attention,
        'multi_head_cross_attention': multi_head_cross_attention
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
    return model, stoi, itos

def generate_caption(model, stoi, itos, image, max_len=30):
    img = tf.image.resize(image, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = tf.expand_dims(img, 0)
    
    encoder_output = model.image_encoder(img)
    output_ids = [stoi["<sos>"]]
    
    for i in range(max_len):
        tgt_input_tokens = tf.expand_dims(output_ids, 0)
        tgt_seq_len = tf.shape(tgt_input_tokens)[1]
        tgt_embeddings = model.tgt_embedding(tgt_input_tokens)
        tgt_embeddings = tgt_embeddings + model.tgt_pos_encoding[tf.newaxis, :tgt_seq_len, :]
        
        dec_out1 = model.decoder1(tgt_embeddings, encoder_output)
        dec_out2 = model.decoder2(dec_out1, encoder_output)
        dec_out3 = model.decoder3(dec_out2, encoder_output)
        
        predictions = model.linear(dec_out3)
        next_token = tf.argmax(predictions[0, -1, :]).numpy()
        
        if next_token == stoi["<eos>"]:
            break
        output_ids.append(int(next_token))
        
    return " ".join(itos[i] for i in output_ids[1:])