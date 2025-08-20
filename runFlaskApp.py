from flask import Flask, request, render_template
import os

app = Flask(__name__)

MODEL_NAMES = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Multinomial Naive Bayes",
    "Linear SVC"
]

loaded_models = {}
global loaded_models
try:
    for name in MODEL_NAMES:
        filename = f'model_{name.replace(" ", "_")}.pkl'
        loaded_models[name] = joblib.load(filename)
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Model files not found! Please run language_detection.ipynb")

vectorizer = joblib.load('language_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    text_input = ""
    selected_model = MODEL_NAMES[0] # Default model = Logistic Regression

    if request.method == 'POST':
        text_input = request.form['text']
        selected_model = request.form['model']
        
        if text_input and selected_model in loaded_models:
            text_vec = vectorizer.transform([text_input]) # Vectorize the user input
            model = loaded_models[selected_model] # Get the selected model
            prediction = model.predict(text_vec)[0] # Predict the language

    return render_template('index.html', 
                           prediction=prediction, 
                           text_input=text_input,
                           models=MODEL_NAMES,
                           selected_model=selected_model)

def create_app_files():
    if not os.path.exists('templates'):
        os.makedirs('templates')

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Language Detector</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
        </style>
    </head>
    <body class="bg-gray-100 text-gray-800 flex items-center justify-center min-h-screen">
        <div class="w-full max-w-2xl mx-auto bg-white rounded-2xl shadow-2xl p-8 md:p-12">
            <h1 class="text-4xl font-bold text-center text-gray-900 mb-2">Language Detector 🌍</h1>
            <p class="text-center text-gray-500 mb-8">Select a model and enter text to identify its language.</p>
            
            <form action="/" method="post" class="space-y-6">
                <div>
                    <label for="model" class="block text-sm font-medium text-gray-700 mb-2">Choose a Model:</label>
                    <select id="model" name="model" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-4 focus:ring-blue-300 focus:border-blue-500 transition duration-300">
                        {% for model in models %}
                            <option value="{{ model }}" {% if model == selected_model %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>
                <textarea name="text" rows="2" class="w-full p-4 border border-gray-300 rounded-lg focus:ring-4 focus:ring-blue-300 focus:border-blue-500 transition duration-300 resize-none" placeholder="Type or paste your text here...">{{ text_input }}</textarea>
                <button type="submit" class="w-full bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition duration-300 text-lg">
                    Detect Language
                </button>
            </form>
            
            {% if prediction %}
            <div id="result" class="mt-10 p-6 bg-gray-50 rounded-lg border border-gray-200 text-center">
                <h2 class="text-xl font-semibold text-gray-800">Detected Language:</h2>
                <p class="text-3xl font-bold text-blue-600 mt-2">{{ prediction }}</p>
                <p class="text-sm text-gray-500 mt-2">Predicted using: {{ selected_model }}</p>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Created 'templates/index.html'.")


if __name__ == '__main__':
    create_app_files()
    print("\nStarting Flask App")
    print("\nOpen your browser and go to http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)