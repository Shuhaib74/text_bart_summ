import os
from dotenv import load_dotenv

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, redirect, render_template, url_for, request, flash, get_flashed_messages, jsonify, session


from text_summ_pre import TextPreprocessor_TextSummarize



# Call the function to load environment variables from a .env file
load_dotenv()

# Creating an instance of Flask
app = Flask(__name__)

# Set the secret key for session management
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'APP_SEC_KEY')

@app.route('/', methods=['GET', 'POST'])
def text_summarization():
    generated_summary = ''
    if request.method == 'POST':
        article = request.form['article']
        device = request.form['device']
        max_len = request.form['max_len']
        min_len = request.form['min_len']
        length_penalty = request.form['len_penalty']

        max_len = int(max_len)
        min_len = int(min_len)
        length_penalty = float(length_penalty)

        summarizer = TextPreprocessor_TextSummarize()

        #
        processed_article = summarizer.preprocess(article)

        # 
        if len(article.split()) <= max_len:
            max_len = len(article.split()) - 50
    
        # Generate summary
        raw_generated_summary = summarizer.generate_summary(processed_article, device, max_len, min_len, length_penalty)

        generated_summary = summarizer.truncate_incomplete_sentence(raw_generated_summary)

        generated_summary = f"Generated Summary: {generated_summary}"
        
    return render_template('text_sum_page.html', generated_summary = generated_summary)




if __name__ == "__main__":
    app.run(debug=True)



