import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from simple_http_server import route, server, Parameter, JSONBody

# Some texts of different lengths.
russian_sentences1 = ["Собака", "Щенки хорошенькие.", "Я люблю гулять вдоль пляжа с собакой."]
russian_sentences2 = ["Собачка"]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Compute embeddings.
ru_res1 = embed(russian_sentences1)
ru_res2 = embed(russian_sentences2)

print(np.inner(ru_res1, ru_res2))
print(np.ravel(np.inner(ru_res1, ru_res2)))

qa = {}
questions = [*qa]
questions_embed = embed(questions)

def make_guess(guess):
    global questions_embed, questions
    guess_embed = embed(guess)
    probabilities = np.ravel(np.inner(questions_embed, guess_embed))
    max_idx = np.argmax(probabilities)
    return questions[max_idx], probabilities[max_idx]

def set_qa(val):
    global qa, questions, questions_embed
    qa = val
    questions = [*qa]
    questions_embed = embed(questions)

@route("/train", method=["POST"])
def train(data=JSONBody([])):
    try:
        set_qa(data)
    except Exception as e:
        return {"success": False}
    return {"success": True}

@route("/guess", method=["POST"])
def guess(data=JSONBody()):
    global qa
    try:
        guess = data.get('guess')
        if not guess:
            return {"success": False}
        best_ans_key, probability = make_guess(guess)
        best_ans = qa[best_ans_key]
        if best_ans and probability > 0.5:
            return {"success": True, "ans": best_ans, "guess": guess}
        return {"success": False, "guess": guess}
    except Exception as e:
        return {"success": False}

server.start(port=8080)