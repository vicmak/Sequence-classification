from nltk.probability import FreqDist
import nltk

def get_tokenized_padded_line(string_line):
    line = string_line.lower()
    line = ExtractAlphanumeric(line)
    tokens = get_padded_sentences_tokens_list(line)
    line = ["<start>"] + tokens + ["<stop>"]

    return " ".join(line)


def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + digits + whitespace + punctuation)])


def get_padded_sentences_tokens_list(text):
    tokens = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent_tokens = nltk.word_tokenize(sent)
        tokens += ["<sentence-start>"] + sent_tokens + ["<sentence-stop>"]

    return tokens


vocab_filename =          "/Users/macbook/Desktop/corpora/MIMIC//MIMIC_50k_vocab.txt"

new_only_notes_filename = "/Users/macbook/Desktop/corpora/MIMIC/notes_discharge_summaries.txt"

counter = 0
all_text = ""

print "Started Reading"

with open(new_only_notes_filename) as f:
    for line in f:

        current_line = get_tokenized_padded_line(line)
        all_text += current_line + " "
        counter += 1
        if counter % 10000 == 0:
            print "Current count:", counter

print "final counter", counter

print "Finished Reading text"

#print all_text

print "Starting tokenization"

tokens = nltk.word_tokenize(all_text)

print "Starting Count"
fdist = FreqDist(tokens)

top_words = fdist.most_common(50000)

print "Writing vocab"

with open(vocab_filename, "w") as vocab_file:
    vocab_file.write("<start> \n")
    vocab_file.write("<stop> \n")
    vocab_file.write("<sentence-start> \n")
    vocab_file.write("<sentence-stop> \n")
    for word_count in top_words:
        vocab_file.write(word_count[0] + "\n")


print "Total:", fdist.N()

print "Top 10", fdist.most_common(10)


