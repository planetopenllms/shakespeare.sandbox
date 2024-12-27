##
#  simple script to analyze input text


from collections import Counter
import matplotlib.pyplot as plt




# construct the training dataset
text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
print(len( text ), text[:100], "..." )

chars = sorted(list(set(text)))
text_size, vocab_size = len(text), len(chars)

print('text has %d characters, %d unique.' % (text_size, vocab_size))
#=>  text has 1_115_394 characters, 65 unique.

count = Counter( text )
print( count )

# Extract characters and their counts for plotting
x = list(count.keys())   # returns iterator?
y = list(count.values())

# Plot the histogram
plt.bar(x, y)
plt.xlabel('Characters')
plt.ylabel('Count')
plt.title('Character Frequency Histogram')
# plt.show()


"""
Counter({' ': 169892, 'e': 94611, 't': 67009, 'o': 65798, 'a': 55507,
'h': 51310, 's': 49696, 'r': 48889, 'n': 48529, 'i': 45537, '\n': 40000,
'l': 33339, 'd': 31358, 'u': 26584, 'm': 22243, 'y': 20448, ',': 19846,
'w': 17585, 'f': 15770, 'c': 15623, 'g': 13356, 'I': 11832, 'b': 11321,
'p': 10808, ':': 10316, '.': 7885, 'A': 7819, 'v': 7793, 'k': 7088,
'T': 7015, "'": 6187, 'E': 6041, 'O': 5481, 'N': 5079, 'R': 4869,
'S': 4523, 'L': 3876, 'C': 3820, ';': 3628, 'W': 3530, 'U': 3313,
'H': 3068, 'M': 2840, 'B': 2761, '?': 2462, 'G': 2399, '!': 2172,
'D': 2089, '-': 1897, 'F': 1797, 'Y': 1718, 'P': 1641, 'K': 1584,
'V': 798, 'j': 628, 'q': 609, 'x': 529, 'z': 356, 'J': 320,
 'Q': 231, 'Z': 198, 'X': 112, '3': 27, '&': 3, '$': 1})
"""


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


### fix $ (only used once e.g.)
#     Now stops thy spring; my sea sha$l suck them dry,



print( chars )
"""
['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?',
 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
"""

print("bye")