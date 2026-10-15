#  USING CONTEXT-FREE DIAGRAM FOR PROPER GRAMMAR IN SIMPLE DECLARATIVE SENTENCES IN HILIGAYNON
import nltk
from nltk.tree import *

grammar = nltk.CFG.fromstring("""
S -> D | E | F | O
A -> a N
B -> b N
C -> c N
D -> d G | d K | d L | d M | d C
E -> e J | e K | e L | e M | e C
F -> f H | f K | f L | f M | f C
G -> g K | g L | g M
H -> h K | h L | h M
I -> i K | i L | i M
J -> j I
K -> k D | k E | k F | k O
L -> l A
M -> m B
N -> n
O -> o K | o L | o M | o C

a -> "tawo" | "balay" | "syudad" | "pagkaon" | "kasapatan" | "kakahuyan" | "baybay" | "hubin" | "pamatyag" | "bulak" | "ilaw" | "asawa" | "gab-i" | "adlaw" | "tubig" | "kapid" | "isda" | "prutas" | "bana" | "utod" | "balasahon" | "kagamitan" | "mangga" | "tinapay" | "lamesa" | "kasilyas" | "kahoy" | "saging" | "nanay" | "abyan" | "telepono" | "kuwarto" | "miga" | "migo"
b -> "hans" | "melmar" | "yves" | "renuel" | "liam" | "junsel" | "milton" | "rene" | "jeremey" | "fredrick" | "anne" | "frank" | "leslie" | "ralph" | "fred" | "vince" | "joel" | "ram" | "randy" | "james"
c -> "sila" | "kita" | "kamo" | "ako" | "ka" | "kami" | "siya"
d -> "naghampang" | "naglumpat" | "nagdalgan" | "nagbaylo" | "naglakat" | "nagtulog" | "nagkaon" | "nagpanaksi" | "nagbasa" | "nag-ampo" | "nagubra" | "nagluto" | "nagtambong" | "nagtukar" | "nagtuon" | "nagpangita" | "nagnegosyo" | "nagdalagan" | "nagtambay" | "nagkasadya" | "naghulat" | "nagkaon" | "nag-um" | "nagbayad" | "nagtulog" | "nagbulong" | "nagtudlo" | "naglantaw" | "nagpadayon" | "nagdumala" | "nag-uli" | "nagpuyo" | "naglayag" | "naglagaw" | "nagpaathag" | "naglantaw" | "naghambal"
e -> "nagahampang" | "nagalumpat" | "nagadalagan" | "nagabaylo" | "nagalakat" | "nagatulog" | "nagakaon" | "nagapanaksi" | "nagabasa" | "nagapangamuyo" | "nagagamit" | "nagaluto" | "nagatambong" | "nagatukar" | "nagatuon" | "nagapangita" | "naganegosyo" | "nagadalagan" | "nagatambay" | "nagakasadya" | "nagahulat" | "nagapangamuyo" | "nagakaon" | "nagapanguroy" | "nagabayad" | "nagatulog" | "nagabulong" | "nagatudlo" | "nagalantaw" | "nagapadayon" | "nagadumala" | "nagpauli" | "nagapuyo" | "nagalayag" | "nagalagaw" | "nagapaathag" | "nagalantaw" | "nagahambal"
f -> "mahampang" | "malumpat" | "madalagan" | "mabaylo" | "malakat" | "matulog" | "makaon" | "mapanaksi" | "mabasa" | "mapangamuyo" | "magamit" | "maluto" | "matambong" | "matukar" | "matuon" | "mapangita" | "magnegosyo" | "madalagan" | "matambay" | "makasadya" | "mahulat" | "mapangamuyo" | "makaon" | "mapanguroy" | "mabayad" | "matulog" | "mabulong" | "matudlo" | "malantaw" | "mapadayon" | "madumala" | "mapauli" | "mapuyo" | "malayag" | "malagaw" | "mapaathag" | "malantaw" | "mahambal"
g -> "kagina" | "kahapon" | "kagab-i" | "sang-una"
h -> "bwas" | "subong" | "karon" | "sadason"
i ->  "mahinay" | "madugay" | "madasig" | "magahod" | "mahipos" | "malayo" | "maluya" | "makasug" | "magrabi" | "maabtik" | "mapagsik"
j -> "sang"
k -> "kag"
l -> "ang"
m -> "si" | "kay"
n -> "."
o -> "guapa" | "law-ay" | "nami" | "mayo" | "maluya" | "matahom" | "maayo" | "mata" | "malimpyo" | "dako" | "mahamot" | "gamay" | "madulom" | "masanag" | "mabaho" | "ka-ilinit" | "mahigko" | "kakadlaw" | "tahom" | "limpyo"
""")

parser = nltk.ChartParser(grammar)

def checkValidity(sts):
  sts = sts.lower()  # Corrected here
  tree_produced = False

  # Attempt to parse the sentence
  for tree in parser.parse(sts.split()):
       #print(tree)
       TREE = Tree.fromstring(str(tree))
       TREE.pretty_print()
       tree_produced = True

  # Check if any trees were produced
  if tree_produced:
      return "valid grammar"
  else:
      return "invalid grammar"

sentence = "guapa si hans . "
print(checkValidity(sentence))
