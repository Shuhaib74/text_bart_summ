
import re 
import zipfile
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def unzip_file(zip_file_path, extract_path):

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# unzip_file(zip_file_path, extract_path)


class TextPreprocessor_TextSummarize:
    def __init__(self):
        self.contractions = {
            "i'll": "i will",
            "ill": "i will",
            "can't": "can not",
            "cant": "can not",
            "cannot": "can not",
            "won't": "will not",
            "wont": "will not",
            "n't": " not",
            "i'm": "i am",
            "im": "i am",
            "it's": "it is",
            "let's": "let us",
            "lets": "let us",
            "you're": "you are",
            "youre": "you are",
            "they're": "they are",
            "we're": "we are",
            "i've": "i have",
            "ive": "i have",
            "you've": "you have",
            "they've": "they have",
            "we've": "we have",
            "i'd": "i would",
            "id": "i would",
            "you'd": "you would",
            "they'd": "they would",
            "we'd": "we would",
            "he'd": "he would",
            "she'd": "she would",
            "it'd": "it would",
            "let's": "let us",
            "'ll": "will",
            "'re": "are",
            "'ve": "have",
            "'d": "would",
            "wanna": "want to",
            "didnt": "did not",
            "dont": "do not",
            "don't": "do not",
            "doesnot": "does not",
            "doesn't": "does not",
            "doesnt": "does not"
        }

    def replace_contractions(self, text):
        for contraction, full_form in self.contractions.items():
            text = re.sub(r'\b{}\b'.format(re.escape(contraction)), full_form, text)
        return text

    def preprocess(self, text):
        text = self.replace_contractions(text)
        return text
    
    def generate_summary(self, article, device, max_len, min_len, length_penalty):
        model_path = './bart_final_model'
        tokenizer_path = './final_tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Initialize summarizer pipeline
        summarizer = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )

        # Generate summary
        generated_summary = summarizer(
            article,
            max_length = max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )[0]['summary_text']

        return generated_summary
        
    def truncate_incomplete_sentence(self, summary):

        pattern = re.search(r'.*[.!?]', summary.strip())
        if pattern:
            return pattern.group(0)
        else:
            return summary
        

if __name__ == "__main__":
    article = '''Mobile gig aims to rock 3G Forget about going to a crowded bar to enjoy a gig by the latest darlings of the music press. Now you could also be at a live gig on your mobile, via the latest third generation (3G) video phones. Rock outfit Rooster are playing what has been billed as the first ever concert broadcast by phone on Tuesday evening from a London venue. The 45minute gig is due to be "phone cast" by the 3G mobile phone operator, 3. 3G technology let us people take, watch and send video clips on their phones, as well as swap data much faster than with 2G networks like GSM. People with 3G phones in the UK can already download football and music clips on their handsets. Some 1,000 fans of the Londonbased band will have to pay five pounds for a ticket and need a 3G handset. "Once you have paid, you can come and go as much as you like, because we expect the customers to be mobile," said 3 spokesperson Belinda Henderson. "Its like going to a concert hall, except that you are virtually there." The company behind the trial hopes to learn more about how people use their video phones. "We are looking on how long people will stay on average on the streams. Some people may stay the whole time, some may dip in and out," said Ms Henderson. "We actually expect people to dip in and out because they are mobile and they will be doing other things." 3 is looking to music as a way of persuading more people to take up the latest video phones. It is already planning regular gigs throughout 2005. And during the intermission, of course, you would still be able to make a phone call.'''

    text_generator = TextPreprocessor_TextSummarize()
    preprocessed_article = text_generator.preprocess(article)
    raw_generated_summary = text_generator.generate_summary(article=preprocessed_article, device='cpu', max_len=200, min_len=50, length_penalty=1.0)
    generated_summary = text_generator.truncate_incomplete_sentence(raw_generated_summary)

    print(generated_summary)
  