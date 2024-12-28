def preprocess_text(text, lang):
    if lang == 'zh':
        # Convert traditional to simplified Chinese
        text = HanziConv.toSimplified(text)
        # Tokenize Chinese text
        return ' '.join(jieba.cut(text))
    elif lang == 'ja':
        japanese_tagger=Tagger('-Owakati')
        return ' '.join([word.surface for word in japanese_tagger(text)])
    elif lang == 'th':
        # Thai tokenization with PyThaiNLP
        return ' '.join(word_tokenize(text))
    else:
        # For other languages, use the original cleaning method
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text