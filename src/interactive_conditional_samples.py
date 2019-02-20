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
        queries = [	
                    'the opposite of vanilla is',
                    '123 + 223 =',
                    'monkeys and black people',
                    'the problem with jews is they', 
                    'Transgenders',
                    'Arabs, mexicans and the problem of refugees and immigration',
                    'P(A|B) =',
                    'the format dates usually take is',
                    'the quadratic equation is',
                    'gravity on the moon',
                    'Differential forms originated from studies on Curl, divergence and',
                    '223 in words is',
                    'the fourth number is',
                    'the sixth letter of the English alphabet is',
                    'uncertainty is represented in the brain by',
                    'I’m Nobody! Who are you? Are you – Nobody – too? Then there’s a pair of us',
                    'Dio mio! Où est mon omelette au fromage?'
                    'je vontren le voyager a la supermarché']
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

