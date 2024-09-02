from flask import Flask, request, render_template, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load model and tokenizer for English to Arabic translation
model_name = 'Helsinki-NLP/opus-mt-en-ar'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form.get('text')
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return jsonify({'translated_text': translated_text[0]})

if __name__ == '__main__':
    app.run(debug=True)