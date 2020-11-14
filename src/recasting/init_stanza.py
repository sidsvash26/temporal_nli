## This script downloads the english package of stanza 
import stanza
stanza.download('en')
stanza_nlp = stanza.Pipeline('en')
