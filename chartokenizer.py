###
#  simple char-level tokenizer


class CharTokenizer:
    def __init__(self, chars):
        self.chars_to_tokens = {t: i for i, t in enumerate(chars)}
        self.tokens_to_chars = {i: t for i, t in enumerate(chars)}
        self.vocab_size = len(chars)

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, text):
        return [self.chars_to_tokens[ch] for ch in text]

    def decode(self, tokens):
        return "".join([self.tokens_to_chars[t] for t in tokens])


if __name__ == '__main__':
    # ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    # ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # digits = '0123456789'
    # punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # whitespace = ' \t\n\r\x0b\x0c'

# 65 unique
# 26 a-z
# 26 A-Z
# -- 52 letters
#  plus punct(uation) and specials
#    13
#  newline (\n) + space ( )
#   .:,;!?
#   $&'-
#   3   - digits (only 3 in  3 KING HENRY VI)


    shakespeare_chars =  'abcdefghijklmnopqrstuvwxyz' + \
                         'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
                         '3' + \
                          '!$&\',-.:;?' + \
                         ' \n'

    print(len(shakespeare_chars), shakespeare_chars)
    # "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ3!$&',-.:;?"

    tokenizer = CharTokenizer( chars=shakespeare_chars )

    start_context = "O God, O God!"
    encoded = tokenizer.encode( start_context )
    print(encoded)
    decoded = tokenizer.decode( encoded )
    print( decoded )

    # encoded = tokenizer.encode( "1234569" )   ## check handling of unknown tokens
    # print( encoded )
    #=>  result in KeyError!!! e.g.  KeyError: '1'

    print("bye")


