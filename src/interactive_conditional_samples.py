#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=3,
    batch_size=None,
    length=None,
    temperature=1,
    top_k=40,
):
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        def run_model(prompt):
            context_tokens = enc.encode(prompt)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    print("\n**Sample " + str(generated) +"**\n")
                    text = enc.decode(out[i])
                    print("*" + prompt + "...*")
                    print(text)
        queries = [  'it is well known that birds are direct descendants of',
                     'the zebra is chasing after the lion because',
                     'the lion is chasing after the zebra because',
                     'the first day of the week is often either',
                     'the opposite of a clean plate is',
                     'the opposite of tall is',  
                     'a synonym for rare is',
                     'a synonym for cromulent is',
                     'a cypher for b-boys and b-girls will',
                     'Despite the similarity between information geometry and the differential geometry as applied in hamiltonian monte carlo, they are quite different because'
                     'In order for a nuclear fusion device to be built, we need to discover ' ]
        for p in queries:
            print('\n## ' + p)
            run_model(p)
            print ('\n---\n')             
        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            print('\n## ' + raw_text)
            run_model(raw_text)
            print("\n---\n")

if __name__ == '__main__':
    fire.Fire(interact_model)

