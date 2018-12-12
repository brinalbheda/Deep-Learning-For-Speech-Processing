## This is a simple tutorial on how to compose the dictionary (lexicon/language - L) together with a language model (grammar - G) similarly to what kaldi needs to do internally for Automatic Speech Recognition
## It will 'create' a dummy text file and a dummy dictionary
## From those it will build the Language L.fst and the Grammar G.fst
## Please go 'quickly' 



. ./path.sh
set -x


echo """
I AY
like L AY K
computer K AH M P Y UW T ER
vision V IH ZH AH N
games G EY M Z
""" > dict
cat dict

sed "s/\s.*//g;s/([0-9])//g" dict  | sort | uniq > wordlist



echo """I like computer
I like vision
I like
like computer
like games
like computer games
like computer vision
I like computer games
I like games
computer games
computer vision games
games like computer vision
I like computer vision 
computer vision
vision like computer
I like computer vision games
""" > text

echo """
<eps> 0
I 1
like 2
computer 3
vision 4
games 5
#0 6
""" > words.syms


ngram-count -order 2 -text text -lm lm.2.arpa
cat lm.2.arpa


echo "Building the FST from the arpa model."
echo "Note that it's not convinient to do that using a script for bigger files so kaldi employs for a piece of this a binary arpa2fst"
echo "but as you did in previous tutorial this isn't anything magic; just faster version of how you did it with the lexicon: build input, output paths that produce words with probabilities "

cat lm.2.arpa | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | \
    fstprint | \
    ./eps2disambig.pl |\
    ./s2eps.pl | \
    fstcompile --isymbols=words.syms \
      --osymbols=words.syms  \
      --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > G.2.fst 

fstdraw  --isymbols=words.syms  --osymbols=words.syms  -portrait G.2.fst | dot -Tpdf > G.2.pdf

# cat lm.2.arpa | \
#     arpa2fst -  \
#     > G.2.bad.fst 

# fstdraw  --isymbols=words.syms  --osymbols=words.syms  -portrait G.2.bad.fst | dot -Tpdf > G.2.bad.pdf


echo """<eps> 0
sil 1
AY 2
L 3
K 4
AH 5
M 6
P 7
Y 8
UW 9
T 10
ER 11
V 12
IH 13
ZH 14
N 15
G 16
EY 17
Z 18
#0 19
""" > phones.syms

echo """
I AY
like L AY K
computer K AH M P Y UW T ER
vision V IH ZH AH N
games G EY M Z
""" > dict_disambig



grep "#"  phones.syms | head -n1  | sed "s/.* //g" >  phones.disambig
grep "#"  words.syms | head -n1  | sed "s/.* //g" >  words.disambig

./make_lexicon_fst.pl dict 0.5 sil '#0' | \
   fstcompile --isymbols=phones.syms \
    --osymbols=words.syms \
    --keep_isymbols=false --keep_osymbols=false |\
   fstarcsort --sort_type=olabel \
   > L.fst

fstdraw  --isymbols=phones.syms  --osymbols=words.syms  -portrait L.fst | dot -Tpdf > L.pdf
