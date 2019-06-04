---
layout: post
title: "PyKomoran을 활용한 자연어처리 뽀개기!"
author: Geunho Lee
description: ""
category: article
tags: [shineware, PyKomoran]
comments: true
---

## PyKomoran을 활용한 자연어처리 뽀개기!

안녕하세요. SHINEWARE의 이근호입니다. 이번 포스트에서는 PyKomoran을 활용해서 여러가지 간단한 자연어처리를 해보려고 합니다. 자연어처리의 범주는 무궁무진한데요. 간단하게는 문서들에서의 단어 빈도수를 구하고 WordCloud 등으로 시각화를 하거나, 긍정 및 부정 문장 분류 등이 있습니다. 이번 포스트에서는 몇 가지 자연어처리를 위해서 형태소 분석 후 관련 예제들을 진행해보겠습니다.

## Data Preparing

형태소 분석을 하기 위한 공개 데이터로 Naver 영화 리뷰 데이터를 읽어옵니다. Pandas는 Web에 있는 문서 읽기도 지원합니다!

```python
import pandas as pd

data_url = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt'
raw_review_data = pd.read_csv(data_url, sep='\t')
raw_review_data.sample(4)
```

id, document, label 3개의 column으로 이루어져 있는 데이터를 확인할 수 있습니다.

| id      | document                                      | label |
| ------- | --------------------------------------------- | ----- |
| 12754   | 사랑의 집착 그 끝은 어디?                     | 1     |
| 3108749 | 어설픈 선정성과 폭력성..                      | 0     |
| 2094352 | 배경음악, 색감, 연기, 구성, 편집 모두 최악. . | 0     |
| 5924993 | 이럴 때 영화가 싫어진다.                      | 0     |

document column을 보시고 눈치채셨겠지만, label 1은 **긍정 리뷰**, 0은 **부정 리뷰**를 의미합니다. 
우리는 document, 즉 Text 데이터를 사용해서 예제를 진행할 것이기 때문에 해당 데이터에 대해서 추가적인 검증을 진행합니다

```python
cleansed_review_data = raw_review_data[~raw_review_data['document'].isnull()]
cleansed_review_data = cleansed_review_data[['document', 'label']]

print('검증 전 레코드 수: {}'.format(len(raw_review_data)))
print('검증 후 레코드 수: {}'.format(len(cleansed_review_data)))
cleansed_review_data.sample(4)
```

```python
검증 전 레코드 수: 200000
검증 후 레코드 수: 199992
```

document 값이 제대로 채워지지 않은 레코드가 8개나 있었군요. 해당 레코드를 버리고, 우리는 필요한 column **document**와 **label** 만 가져와서 사용하겠습니다. 여기까지가 데이터 준비단계였습니다.

## Data Preprocessing

형태소 빈도 분석, 문장 분류 등 여러 가지 자연어처리 작업을 위해서는 사전에 Data Preprocessing이 필요합니다. 이번 포스트에서는 PyKomoran을 활용해서 데이터 전처리를 해보겠습니다. PyKomoran은 pip를 통해서 쉽게 설치가 가능합니다!

```shell
pip install PyKomoran
```

설치 후 PyKomoran을 불러오겠습니다

```python
from PyKomoran import *
komoran = Komoran('STABLE')
```

Komoran 사용해서 앞에서 준비한 데이터에 대해서 형태소 분석을 해보겠습니다.

```python
cleansed_review_data['tokens'] = cleansed_review_data['document'].map(lambda s: komoran.get_plain_text(s).split('\s'))
cleansed_review_data.sample(4)
```

| document                                                     | label | tokens                                                       |
| ------------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| 7~8점정도는 되는데 현실왜곡해서 1점줌. 마지막에 말이되냐. 주인공 버프 개사기.... | 0     | [7/SN ~/SO 8/SN 점/NNB 정도/NNG 는/JX 되/VV 는데/EC 현...    |
| 지루함+어수선함=싸이코패스 영화                              | 0     | [지루/XR 하/XSA ㅁ/ETN +/SW 어수선/XR 하/XSA ㅁ/ETN =/S...   |
| 속 깊은 진짜 사나이. 그랜토리노!                             | 1     | [속/NNG 깊/VA 은/ETM 진짜/MAG 사나이/NNG ./SF 그러/VV 어/... |
| 원작을 보지 못해서 내용의 충실도에 대해선 뭐라 할 순 없지만 영화가 흘러가는 완급... | 1     | [원작/NNG 을/JKO 보/VV 지/EC 못하/VX 아서/EC 내용/NNG 의/J... |

tokens column을 통해 문장이 형태소로 분해된 것을 확인할 수 있습니다. 형태소 각각에 대한 품사표는 https://pydocs.komoran.kr/firststep/postypes.html 를 참고하세요! 위 예제를 봤을 때 특수 문자나, 몇가지 형태소는 우리의 관심사에서 조금 동떨어진 것 같습니다. PyKomoran에서는 사용자가 관심 있는 형태소에 대해서만 추출이 가능합니다. 저는 명사, 동사 및 형용사 형태소만 따로 뽑아보겠습니다.

```python
# 순서대로 일반 명사, 동사, 형용사
target_tags = ['NNG','VV', 'VA']
cleansed_review_data['specific_tokens'] = cleansed_review_data['document'].map(lambda s: komoran.get_morphes_by_tags(s, tag_list=target_tags))
cleansed_review_data.sample(4)[['document', 'label', 'specific_tokens']]
```

| document                                                     | label | specific_tokens                                              |
| ------------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| 이 영화는진짜별점으로 매길수없는영화입니다                   | 1     | [영화, 진짜, 별점, 매기, 없, 영화]                           |
| 일본판 쏘우를 기대했는데 실망                                | 0     | [쏘, 기대, 실망]                                             |
| 좀 아쉬운                                                    | 0     | [아쉽]                                                       |
| 정말 유치함재난영화에 대한 예의로 다봤을뿐중간에 헛웃음 계속 나오는 영화애들은 계속... | 0     | [재난, 영화, 대하, 예의, 보, 중간, 웃음, 나오, 영화, 애, 발암, 유발] |

**specific_tokens** column을 통해 처음에 지정한 일반 명사, 동사, 형용사 형태소만 뽑은 것을 확인할 수 있습니다. 이제 여기까지 전처리가 끝난 데이터를 가지고 여러 가지 작업을 해보겠습니다.

## 형태소 빈도 분석

전처리가 끝난 데이터를 통해서 긍정 리뷰와 부정 리뷰에서 자주 등장하는 형태소 빈도 분석을 해보겠습니다. 이번 포스트에서는 일반 명사, 동사, 형용사 형태소에 대해서 빈도 분석을 해보겠습니다.

```python
import itertools

positive_review_data = cleansed_review_data[cleansed_review_data['label'] == 1]
negative_review_data = cleansed_review_data[cleansed_review_data['label'] == 0]

positive_specific_tokens = list(itertools.chain.from_iterable(positive_review_data['specific_tokens']))
negative_specific_tokens = list(itertools.chain.from_iterable(negative_review_data['specific_tokens']))
```

Python에서 제공하는 Counter를 사용하여 쉽게 빈도수를 계산할 수 있습니다.

```python
from collections import Counter

print("긍정 리뷰 형태소 (일반 명사 / 동사 / 형용사)")
Counter(positive_specific_tokens).most_common()[:30]
```

```
긍정 리뷰 형태소 (일반 명사 / 동사 / 형용사)
```

```
[('영화', 38823),
 ('보', 31835),
 ('좋', 11156),
 ('재밌', 9271),
 ('하', 9085),
 ('있', 8744),
 ('최고', 7760),
 ('없', 6239),
 ('같', 5341),
 ('연기', 5105),
 ('감동', 4889),
 ('되', 4852),
 ('때', 4471),
 ('생각', 4456),
 ('재미있', 4416),
 ('드라마', 3871),
 ('만들', 3648),
 ('사랑', 3643),
 ('평점', 3517),
 ('나오', 3506),
 ('사람', 3288),
 ('말', 3210),
 ('배우', 3067),
 ('나', 2820),
 ('재', 2748),
 ('스토리', 2538),
 ('알', 2537),
 ('작품', 2446),
 ('마지막', 2407),
 ('명작', 2294)]
```

긍정 리뷰에서는 자주 나왔던 형태소(일반 명사, 동사, 형용사)를 순서대로 세워보았습니다. 긍정의 표현인 재밌, 최고, 감동, 명작 등의 형태소가 눈에 띄네요.

```python
from collections import Counter

print("부정 리뷰 형태소 (일반 명사 / 동사 / 형용사)")
Counter(negative_specific_tokens).most_common()[:30]
```

```
부정 리뷰 형태소 (일반 명사 / 동사 / 형용사)
```

```
[('영화', 36977),
 ('보', 23975),
 ('없', 14467),
 ('하', 11811),
 ('만들', 6390),
 ('같', 5474),
 ('나오', 5384),
 ('아깝', 5146),
 ('있', 5057),
 ('평점', 4810),
 ('스토리', 4569),
 ('연기', 4363),
 ('재미없', 4312),
 ('좋', 4266),
 ('쓰레기', 4254),
 ('내용', 3885),
 ('감독', 3695),
 ('시간', 3690),
 ('말', 3604),
 ('되', 3466),
 ('안', 3458),
 ('사람', 3325),
 ('배우', 3322),
 ('재미', 3150),
 ('나', 3064),
 ('알', 3040),
 ('최악', 2910),
 ('이렇', 2807),
 ('드라마', 2751),
 ('생각', 2598)]
```

부정 리뷰에서는 아깝, 재미없, 쓰레기, 최악 등의 형태소가 눈에 띕니다. 빈도 분석을 텍스트로만 하니까 조금 심심하네요. 워드클라우드를 활용해서 시각화를 해보겠습니다.

## 형태소 빈도 시각화

이번 포스트에서는 Python의 wordcloud 패키지로 시각화를 해보겠습니다. 혹시 설치가 되어있지 않다면 pip로 쉽게 설치할 수 있습니다.

```sh
pip install wordcloud
```

WordCloud 객체애 대해서 다음과 같이 설정을 합니다. 한글이 들어가기 때문에 폰트 path에 대해서 지정을 해줘야 깨지지 않고 볼 수 있습니다. 아래 세팅은 https://lovit.github.io/nlp/2018/04/17/word_cloud/ 에서 많이 참고하였습니다.

```python
from wordcloud import WordCloud

wordcloud = WordCloud(
    font_path='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
    width=800,
    height=800,
    background_color='white'
)
```

긍정 리뷰에 대해서 먼저 시각화해보겠습니다

```python
%matplotlib inline
import matplotlib.pyplot as plt
wordcloud = wordcloud.generate_from_frequencies(Counter(positive_specific_tokens))

fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud.to_array(), interpolation="bilinear")
plt.show()
```

![positive_cloud](/images/positive_cloud.png?raw=true)

영화 리뷰다보니 영화가 많이 등장하였고 그 외에 최고, 감동, 재밌 등이 보입니다.
부정 리뷰에 대해서도 시각화를 해보겠습니다.

```python
%matplotlib inline
import matplotlib.pyplot as plt
wordcloud = wordcloud.generate_from_frequencies(Counter(negative_specific_tokens))

fig = plt.figure(figsize=(10, 10))
plt.imshow(wordcloud.to_array(), interpolation="bilinear")
plt.show()
```

![negative_cloud](/images/negative_cloud.png?raw=true)

마찬가지로 영화 키워드가 많이 보이며, 그 외 최악, 재미없 등이 보입니다.
