# 10 Failure Cases — Stratified Across Configurations

Each case is the lowest-F1 EM=0 row from a distinct (model, dataset, retrieval, k) combination, to surface a range of failure modes rather than concentrating on a single weak config.


## Case 1 — qwen · trivia_qa · none · k=0
- **Question:** In the TV sitcom Adam's Rib, who played the Spencer Tracy Film role?
- **Gold (first):** Ken Howard
- **Prediction:** `Spencer Tracy was played by Henry Fonda, and Katharine Hepburn was played by Spencer Tracy.`
- **Top retrieved doc:** _(none — closed-book run)_
- **EM=0, F1=0.000**

## Case 2 — qwen · trivia_qa · bm25 · k=5
- **Question:** In which state is Camp David?
- **Gold (first):** Maryland
- **Prediction:** `I don't know.`
- **Top retrieved doc:** Dalai Camp
- **EM=0, F1=0.000**

## Case 3 — phi3 · natural_questions · bm25 · k=1
- **Question:** where does a hamster live in the wild
- **Gold (first):** Hamster Hamsters are more crepuscular than nocturnal and, in the wild, remain underground during the day to avoid being caught by predators. They feed primarily on seeds, fruits, and vegetation, and will occasionally eat burrowing insects.[3] As one of their more prominent characteristics, they hav…
- **Prediction:** `I don't know.`
- **Top retrieved doc:** The Call of the Wild The Call of the Wild is a short adventure novel by Jack London published in 1903 and set in Yukon, Canada during the 1890s Klondike Gold Rush, when strong sled dogs were in high demand. The central character of the novel is a dog named Buck. The story opens at a ranch in Santa …
- **EM=0, F1=0.000**

## Case 4 — phi3 · natural_questions · bm25 · k=10
- **Question:** when did the us military start using hummers
- **Gold (first):** Humvee In 1979, the U.S. Army drafted final specifications for a High Mobility Multipurpose Wheeled Vehicle (HMMWV), which was to replace all the tactical vehicles in the 1/4 to 1 1/4-ton range,[8] namely the M151 quarter-ton jeep and M561 Gama Goat, as one "jack-of-all-trades" light tactical vehic…
- **Prediction:** `I don't know.`
- **Top retrieved doc:** Five-star rank Five-star ranks were created in the US military during World War II because of the awkward situation created when some American senior commanders were placed in positions commanding allied officers of higher rank.[8] US officers holding five-star rank never retire; they draw full act…
- **EM=0, F1=0.000**

## Case 5 — qwen · trivia_qa · dpr_nq · k=5
- **Question:** In Top Cat, who was the voice of Choo Choo?
- **Gold (first):** Marvin Kaplan
- **Prediction:** `I don't know.`
- **Top retrieved doc:** America, United States of
- **EM=0, F1=0.000**

## Case 6 — phi3 · natural_questions · bm25 · k=5
- **Question:** who gave the motto back to the vedas
- **Gold (first):** Dayananda Saraswati Dayananda's "back to the Vedas" message influenced many thinkers and philosophers the world over.[21]
- **Prediction:** `I don't know.`
- **Top retrieved doc:** Jarasandha Jarasandha's father king Brihadratha was married to the twin daughters of the King of Kashi. Brihadratha loved both his wives equally, but had no sons. Once sage Chandakaushika visited his kingdom and gave a mango to the king as a boon. The king divided the mango equally and gave to his …
- **EM=0, F1=0.000**

## Case 7 — qwen · trivia_qa · dpr_nq · k=10
- **Question:** "Who did Mrs. Thatcher describe as ""a man we can do business with?"""
- **Gold (first):** Gorbachev
- **Prediction:** `I don't know.`
- **Top retrieved doc:** Sculptor (profession)
- **EM=0, F1=0.000**

## Case 8 — qwen · trivia_qa · dpr_multiset · k=1
- **Question:** Who was the first MVP in a Super Bowl to be on the losing side?
- **Gold (first):** Chuck Howley
- **Prediction:** `I don't know.`
- **Top retrieved doc:** Kramer vs. Kramer
- **EM=0, F1=0.000**

## Case 9 — phi3 · natural_questions · dpr_nq · k=1
- **Question:** who sang what in the world's come over you
- **Gold (first):** Jack Scott (singer) At the beginning of 1960, Scott again changed record labels, this time to Top Rank Records.[1] He then recorded four Billboard Hot 100 hits – "What in the World's Come Over You" (#5), "Burning Bridges" (#3) b/w "Oh Little One" (#34), and "It Only Happened Yesterday" (#38).[1] "W…
- **Prediction:** `I don't know.`
- **Top retrieved doc:** It Ain't Over 'til It's Over "It Ain't Over 'til It's Over" is a song recorded, written, and produced by American musician Lenny Kravitz for his second studio album, Mama Said (1991). Released as the album's second single in June 1991, the song is a mid-tempo ballad, musically inspired by Motown, P…
- **EM=0, F1=0.000**

## Case 10 — phi3 · natural_questions · dpr_nq · k=5
- **Question:** what part of the country are you likely to find the majority of the mollisols
- **Gold (first):** Mollisol Mollisols occur in savannahs and mountain valleys (such as Central Asia, or the North American Great Plains). These environments have historically been strongly influenced by fire and abundant pedoturbation from organisms such as ants and earthworms. It was estimated that in 2003, only 14 …
- **Prediction:** `I don't know.`
- **Top retrieved doc:** Malaysia Malaysia (/məˈleɪʒə/ ( listen) mə-LAY-zhə or /məˈleɪsiə/ ( listen) mə-LAY-see-ə; Malaysian pronunciation: [məlejsiə])[lacks stress] is a federal constitutional monarchy located in Southeast Asia. It consists of thirteen states and three federal territories and has a total landmass of 330,8…
- **EM=0, F1=0.000**
